import argparse
from pathlib import Path
import pickle
import json
from copy import deepcopy

import numpy as np
import rasterio
from pyproj import Transformer
import cv2
from skimage import color
from skimage.feature import local_binary_pattern
from tqdm import tqdm
from fastprogress.fastprogress import master_bar, progress_bar

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score

from shadows.shadowcorrect.RegionGroupMatching import RegionGroupMatching
from shadows.shadowcorrect.utils import apply_MeanShift_clustering, get_image_and_mask
from shadows.shadowcorrect.run_model import config


def get_clustering_features_from_image(image, config_file):
    """Extract features from image for clustering.

    Features are the LocalBinaryPattern taken from the image, hsv and lab spaces
    
    :param image: array. RGB image
    
    :returns array with 9 features"""
    lab_image = image / 255.0
    lab_image = color.rgb2lab(lab_image).astype(int)
    
    # Convert to HSV color space
    hsv_image = image / 255.0
    hsv_image = color.rgb2hsv(hsv_image)    
    hsv_image[..., 0] = hsv_image[..., 0] * 360 # Multiply the H channel by 360 to convert it to [0, 360] range
    hsv_image[..., 1] = hsv_image[..., 1] * 100 # Multiply the S channel by 100 to convert it to [0, 100] range
    hsv_image[..., 2] = hsv_image[..., 2] * 100 # Multiply the V channel by 100 to convert it to [0, 100] range
    hsv_image = hsv_image.astype(int)

    # make feature vector
    vector = []
    for img in [image, hsv_image, lab_image]:                            
        # Split channels
        channels = cv2.split(img)
        for channel in channels:
            lbp = local_binary_pattern(channel, P=config_file['n_points'], R=config_file['radius'], method='ror')
            vector.append(np.mean(lbp))
    return np.array(vector)


def cluster_images(image_dir, config_file):
    """
    Clusters images based on their color and texture features using K-Means clustering.
    :param image_dir: Path, Directory containing the input images. Each image should have at least 3 channels (e.g., RGB).
    :param config_file: dict, Configuration dictionary.
    :returns: list of str, ndarray
    """
    all_images = [str(f) for f in image_dir.iterdir() if f.is_file()]

    seed = config_file['seed']
    
    mean_eigenvectors = {}
    for image_path in tqdm(all_images):    
        with rasterio.open(image_path) as src:
            R = src.read(1)
            G = src.read(2)
            B = src.read(3)
        image = np.dstack([R, G, B])

        mean_eigenvectors[image_path] = get_clustering_features_from_image(image, config_file)

    keys = list(mean_eigenvectors.keys())
    data = np.array(list(mean_eigenvectors.values()))
    
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data)

    # Extract MinMax statistics
    scaling_stats = {
        "min_": scaler.min_.tolist(),  # Feature-wise minimums
        "scale_": scaler.scale_.tolist(),  # Feature-wise scaling factors
        "feature_range": scaler.feature_range,  # The scaling range (default is (0, 1))
    }
    
    # Range for k
    k_values = range(2, config_file['clustering_optimization_max_K_clusters'])  # Test k from 2 to clustering_optimization_max_K_clusters
    ch_scores = []
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, n_init='auto', random_state=seed)
        labels = kmeans.fit_predict(data_normalized)
        
        # Calculate Calinski-Harabasz score
        score = calinski_harabasz_score(data_normalized, labels)
        ch_scores.append(score)
    
    # Find the best k
    best_k = k_values[np.argmax(ch_scores)]
    print(f"Best k: {best_k}")
    
    kmeans = KMeans(n_clusters=best_k, n_init='auto', random_state=seed)
    labels_ = kmeans.fit_predict(data_normalized)
    labels = {k: l for k, l in zip(keys, labels_)}

    centroids = kmeans.cluster_centers_
    
    return labels, scaling_stats, centroids


def optimize_image_parameters(image, mask, image_latlon, image_acquisition_time, config_file):
    """Optimize RegionGroupMatching parameters for image
    
    :param image: 3D array. Raw image
    :param mask: 2D array. Shadow mask
    :param image_acquisition_time: str. Timestamp of acquisition, if available
    :param image_latlon: tuple. (latitude, longitude) coordinates of image center

    :return best_light_params, best_shadow_params
    """
    # Perform RGM
    rgm = RegionGroupMatching(original_image=image, shadow_mask=mask, image_latlon=image_latlon, image_acquisition_time=image_acquisition_time)
    rgm.initial_overall_shadow_removal(kernel_sigma=config_file['kernel_sigma'], exposure_factor=1)
    
    # Convert to LAB color space
    lab_image = cv2.GaussianBlur(rgm.initial_corrected_image, (5, 5), 0)
    lab_image = cv2.cvtColor(lab_image, cv2.COLOR_RGB2LAB)

    # Create an empty copy of the image for each region
    shadow_regions = np.zeros_like(lab_image)  # Image where mask == 1
    light_regions = np.zeros_like(lab_image)  # Image where mask == 0

    # Apply the mask to create the two images
    shadow_regions[rgm.shadow_mask == 1] = lab_image[rgm.shadow_mask == 1]  # Keep pixels where mask is 1
    light_regions[rgm.shadow_mask == 0] = lab_image[rgm.shadow_mask == 0]  # Keep pixels where mask is 0
    
    # Group shadow regions    
    _, best_shadow_params = apply_MeanShift_clustering(image=shadow_regions, mask=rgm.shadow_mask, param_finetuning=config_file['clustering_optimization_shadow_finetuning'], verbose=False)
            
    # Group light regions
    _, best_light_params = apply_MeanShift_clustering(image=light_regions, mask=(1 - rgm.shadow_mask), param_finetuning=config_file['clustering_optimization_light_finetuning'], verbose=False)

    return best_light_params, best_shadow_params
    

def find_optimal_parameters(labels, shadow_mask_dir, config_file):
    """
    Finds the optimal MeanShift segmentation parameters for shadow and light regions in clustered images.
    :param labels: dict, Dictionary with image files as keys and cluster labels as values.
    :param shadow_mask_dir: Path, Directory containing shadow mask files (.npy) corresponding to the images.
    :param config_file: dict, Configuration dictionary.
    :param save_fname: str, optional, Filename to save the results. Defaults to 'optimal_MeanShift_parameters' if not specified.
    """
    # Initialize a dictionary to store image names by cluster
    clustered_images = {}  
    # Iterate over the labels and image names
    for img_name, cluster in labels.items():
        # Initialize a list for the cluster if it doesn't exist and add current image
        clustered_images[cluster] = clustered_images.get(cluster, []) + [img_name]

    np.random.seed(config_file['seed'])
    params_results = {}
    mb = master_bar(list(set(labels.values())))
    for cluster_index in mb:
        mb.main_bar.comment = f'Optimizing images of cluster {cluster_index}'
        params_results[cluster_index] = {}
        print(f'Applying MeanShift on {config_file["clustering_optimization_max_images"]} images from cluster {cluster_index}.')

        # define max number of images for parameter optimization of cluster label
        max_images = (len(clustered_images[cluster_index])
                      if config_file['clustering_optimization_max_images'] > len(clustered_images[cluster_index])
                      else config_file['clustering_optimization_max_images'])

        # select random images for testing
        selected_images = np.random.choice(clustered_images[cluster_index],
                                           size=max_images,
                                           replace=False)
        
        for img_name in progress_bar(selected_images, parent=mb):
            mb.child.comment = f'Optimizing image {Path(img_name).stem}'
            # retrieve image, mask and image parameters for image file
            image, mask, image_acquisition_time, image_latlon, _ = get_image_and_mask(
                Path(img_name), shadow_mask_dir)
            
            if len(np.unique(mask)) == 2:
                params_results[cluster_index][img_name] = {}

                # optimize light and shadow parameters
                best_light_params, best_shadow_params = optimize_image_parameters(
                    image, mask, image_latlon, image_acquisition_time, config_file)
                # best_shadow_params = {"best_spatial_radius_value": 12, "best_range_radius": 9.0, "best_min_density_value": 130}
                # best_light_params = {"best_spatial_radius_value": 12, "best_range_radius": 9.0, "best_min_density_value": 130}
                params_results[cluster_index][img_name]['shadow_params'] = best_shadow_params
                params_results[cluster_index][img_name]['light_params'] = best_light_params
            
    # Filtered dictionary
    filtered_params_dict = {}
    for cluster_index, p_dict in params_results.items():
        tmp_filtered_params_dict = {}
        for key, p_value in p_dict.items():
            # check if any param value is 0
            has_zeros = np.any(
                np.array([list(p_value[v].values())
                          for v in ['shadow_params', 'light_params']]).flatten() == 0)
            if not has_zeros:
                tmp_filtered_params_dict[key] = p_value
        filtered_params_dict[cluster_index] = tmp_filtered_params_dict

    # aggregation function for best_range_radius parameter
    def agg_best_range_radius(x):
        mean_x = np.mean(x)
        rounded_x = (np.floor(mean_x)
                        if mean_x - np.floor(mean_x) < 0.5
                        else np.ceil(mean_x))
        return rounded_x
        
    agg_funs = {'best_range_radius': agg_best_range_radius,
                'best_spatial_radius_value': lambda x: int(np.mean(x)),
                'best_min_density_value': lambda x: int(np.mean(x))}
    # Print stats
    final_params = {}
    for cluster_index, cluster_data in filtered_params_dict.items():
        final_params[int(cluster_index)] = {'shadow_params': {}, 'light_params': {}}

        param_values = {k: {} for k in ['shadow_params', 'light_params']}
        
        # Loop through the input dictionary to populate the output
        for image_data in cluster_data.values():
            for param_type in ["shadow_params", "light_params"]:
                for param, value in image_data[param_type].items():
                    param_values[param_type][param] = param_values[param_type].get(param, [])
                    param_values[param_type][param].append(value)

        # Loop through each parameter type and parameter to calculate aggregate cluster values
        for param_type, param_value in param_values.items():
            for param, values in param_value.items():
                agg_param_value = agg_funs[param](values)
                final_params[cluster_index][param_type][param] = agg_param_value
                print(f"  {param_type} {param}: Mean: {agg_param_value:.2f}, "
                      f"Min: {np.min(values)}, Max: {np.max(values)}")

    return final_params
    
if __name__ == "__main__":
    # ------------- Parse input arguments
    parser = argparse.ArgumentParser(description='Cluster images and find the best parameters for segmentation for shadow correction')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory with image data')
    parser.add_argument('--shadow_mask_dir', 
                        type=str, 
                        required=True, 
                        help='Directory with shadow mask data (mask files must have same name as image files)')
    parser.add_argument('--output_dir', 
                        type=str, 
                        required=True, 
                        help='Directory where to store cluster labels and parameters configuration')
    parser.add_argument('--config_path', 
                        type=str, 
                        required=False, 
                        help='Path for custom configuration json file. see run_model.config for options')
    
    args = parser.parse_args()
    
    # read config file
    config_file = deepcopy(config)
    if args.config_path is not None:
        if Path(args.config_path).exists():
            config_ = json.load(open(args.config_path, 'rb'))
            config_file.update(config_)
        else:
            raise ValueError(f'{args.config_path} does not exist')
        
    # Cluster images
    labels, scaling_stats, centroids = cluster_images(image_dir=Path(args.image_dir), 
                                                      config_file=config_file)

    # Find best parameters for each cluster
    optimized_cluster_parameters = find_optimal_parameters(
        labels, shadow_mask_dir=Path(args.shadow_mask_dir), 
        config_file=config_file)

    output = config
    output.update({'optimized_cluster_parameters': optimized_cluster_parameters,
                   'cluster_centroids': centroids.tolist(),
                   'scaling_stats': scaling_stats})
    print(output)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'optimized_cluster_parameters.json', 'w') as config_filename:
        json.dump(output, config_filename)

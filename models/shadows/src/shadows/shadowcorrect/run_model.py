import argparse
from pathlib import Path
from collections import Counter
import os
import json
import rasterio
from copy import deepcopy
from tqdm import tqdm

import numpy as np
import random

import cv2
from sklearn.metrics import pairwise_distances_argmin_min
from skimage import color
from skimage.feature import local_binary_pattern

from shadows.shadowcorrect.RegionGroupMatching import RegionGroupMatching
from shadows.shadowcorrect.utils import get_image_and_mask


config = {'n_points': 2,
          'radius': 12,
          'clustering_optimization_max_K_clusters': 11,
          'clustering_optimization_max_images': 50,
          'clustering_optimization_shadow_finetuning': {
             'spatial_radius_values': [4, 6, 8, 10, 12],
             'range_radius_values': [5.0, 7.0, 9.0, 11.0, 13.0, 15.0],
             'min_density_values': [90, 110, 130, 150, 170, 190]},
          'clustering_optimization_light_finetuning': {
              'spatial_radius_values': [6, 8, 10, 12, 14, 16],
              'range_radius_values': [9.0, 11.0, 13.0, 15.0, 17.0, 19.0],
              'min_density_values': [110, 130, 150, 170, 190, 210]},
          'kernel_sigma': 10,
          'exposure_factor': 4,
          'verbose': False,
          'gradient_kernel_size': 3,
          'threshold': 0.2,
          'spatial_weight': 1.0,
          'pixel_value_weight': 10.0,
          'lambda_factor': 0.5,
          'shadow_finetuning': {'spatial_radius_values': [10], 'range_radius_values': [13.0], 'min_density_values': [150]},
          'light_finetuning': {'spatial_radius_values': [12], 'range_radius_values': [9.0], 'min_density_values': [190]}, 
          'seed': 666}


def get_image_paths(image_dir):
    """
    Prepares input data by collecting file paths for images and corresponding shadow masks.
    :param image_dir: str or Path, Directory containing the input images.
    :returns: List of file paths to the image files.
    """
    image_paths = list(Path(image_dir).glob('*'))

    if len(image_paths) == 0:
        raise ValueError('No images found in directory')
        
    return image_paths


def save_corrected_image(image, output_dir, image_name, metadata):
    """ Saves shadow-corrected image as a tif file with original metadata updated

    :param image: array. RGB image as a numpy array
    :param output_dir: pathlib. Path to directory where to store shadow-corrected image
    :param image_name: str. Name of image without extension
    :param metadata: dict. Dictonary with original tif image's metadata
    """
    # Update the number of channels to 3
    metadata.update({"count": 3, "width": image.shape[1], "height": image.shape[0]})
    rgb_data = (image / np.max(image) * 255).astype(np.uint8)
    rgb_data = rgb_data.transpose(2, 0, 1)
    
    with rasterio.open(output_dir / f'{image_name}.tif', "w", **metadata) as dst:
        dst.write(rgb_data)
    

def correct_shadows(image_paths, shadow_mask_dir, config_mode, config_file, output_dir, save_fname='', recompute_if_exists=True):
    """
    Corrects shadows in images using a multi-step shadow removal process and calculates evaluation metrics.
    :param image_paths: list, containing file paths to the image files.
    :param shadow_mask_dir: pathlib. Directory with shadow mask data (mask files must have same name as image files)
    :param config_mode: str, Mode for configuring shadow removal parameters. 
    :param config_file: dict, Configuration dictionary 
    :param output_dir: Path, Directory where corrected images and outputs will be stored.
    :param save_fname : str, optional, Filename to save evaluation metrics. Defaults to 'metrics' if not specified.
    :param recompute_if_exists: bool. Whether to re-correct image if shadow-corrected file exists on disk. Default=False
    """
    
    metrics = {}
    for ix, image_path in enumerate(tqdm(image_paths)):
        output_file = output_dir / f'{Path(image_path).stem}.tif'
        if (output_file.exists()) & (recompute_if_exists is False):
            print(f'Shadow-corrected image {output_file.stem} exists, skipping...')
            continue
        
        image, mask, image_acquisition_time, image_latlon, metadata = get_image_and_mask(
                image_path, shadow_mask_dir)

        n_classes_mask = len(np.unique(mask))
        if n_classes_mask == 1:
            print('Only one class found in shadow mask, saving original image')
            save_corrected_image(image, output_dir, f'{Path(image_path).stem}', metadata)
        elif n_classes_mask != 2:
            raise ValueError(f'Shadow mask of {Path(image_path).stem} has {n_classes_mask} classes')
        else:
            # Region Group Matching
            rgm = RegionGroupMatching(original_image=image, 
                                      shadow_mask=mask, 
                                      image_latlon=image_latlon, 
                                      image_acquisition_time=image_acquisition_time)
    
            # ----- Step 1 -----
            rgm.initial_overall_shadow_removal(kernel_sigma=config_file['kernel_sigma'], 
                                               exposure_factor=config_file['exposure_factor'])
    
            # ----- Step 2 -----
            shadow_finetuning, light_finetuning = get_rgm_parameters(
                config_mode, config_file=config_file, image=image)
                
            rgm.internal_grouping_of_shadow_and_light_regions(
                shadow_finetuning=shadow_finetuning, 
                light_finetuning=light_finetuning, 
                verbose=config_file['verbose'])
    
            # ----- Step 3 -----
            rgm.feature_extraction_and_matching(
                radius=config_file['radius'], 
                n_points=config_file['n_points'], 
                gradient_kernel_size=config_file['gradient_kernel_size'])
    
            # ----- Step 4 -----
            rgm.local_shadow_region_enhancement(threshold=config_file['threshold'], 
                                                verbose=config_file['verbose'])
    
            # ----- Step 5 -----
            rgm.shadow_boundary_optimization(kernel_size=config_file['gradient_kernel_size'], 
                                             spatial_weight=config_file['spatial_weight'], 
                                             pixel_value_weight=config_file['pixel_value_weight'], 
                                             lambda_factor=config_file['lambda_factor'])
    
            # Save image
            save_corrected_image(rgm.final_corrected_image, output_dir, f'{Path(image_path).stem}', metadata)


def get_rgm_parameters(config_mode, config_file=None, image=None):
    """Gets (optimized) RegionalGroupMatching parameters

    :params config_mode: str. Can be 'manual'/'default' (read from config file) or 
    'auto', in which case the image will be assigned to its nearest precomputed cluster 
    centroids and their optimized parameters will be used. 
    :param config_file: dict. Configuration file with optimized parameters. If config_mode='manual', 
    cannot be None. Default=None
    :param image: array. RGB image. Default=None. If config_mode == 'auto', cannot be None

    :return light_finetuning: dictionary with optimized parameters for light finetuning
    :return shadow_finetuning: dictionary with optimized parameters for shadow finetuning
    """
    from shadows.shadowcorrect.cluster_images import get_clustering_features_from_image
    if config_mode in ['default', 'manual']:
        if config_file is None:
            raise ValueError("if config_mode is 'manual', config_file cannot be None")
        shadow_finetuning = config_file['shadow_finetuning']
        light_finetuning = config_file['light_finetuning']
        return light_finetuning, shadow_finetuning
        
    elif config_mode == 'auto':
        if image is None:
            raise ValueError("if config_mode is 'manual', image cannot be None")
        # extract LBP features for clustering
        new_data = get_clustering_features_from_image(image, config_file).reshape(1, -1)

        # scale data using stored statistics
        min_ = np.array(config_file['clustering_parameter_optimization']['scaling_stats']["min_"])
        scale_ = np.array(config_file['clustering_parameter_optimization']['scaling_stats']["scale_"])
        
        # Apply scaling manually
        new_data_normalized = (new_data - min_ / scale_)

        # get cluster centroids
        centroids = np.array(config_file['clustering_parameter_optimization']['cluster_centroids'])

        # identify nearest cluster
        closest_cluster, _ = pairwise_distances_argmin_min(new_data_normalized, centroids)

        # get parameters of nearest cluster
        params_closest_cluster = config_file['clustering_parameter_optimization']['optimized_cluster_parameters'][str(closest_cluster[0])].copy()
        param_mapper = {'best_spatial_radius_value': 'spatial_radius_values',
                        'best_range_radius': 'range_radius_values',
                        'best_min_density_value': 'min_density_values'}
        [{v: params_closest_cluster[f'{scope}_params'][k] for k, v in param_mapper.items()} for scope in ['light', 'shadow']]
        shadow_finetuning, light_finetuning = [{v: params_closest_cluster[f'{scope}_params'][k]
                                                for k, v in param_mapper.items()} for scope in ['light', 'shadow']]

        return shadow_finetuning, light_finetuning

if __name__ == "__main__":
    # ------------- Parse input arguments
    parser = argparse.ArgumentParser(description='Run Shadow Correction')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory with image data')
    parser.add_argument('--shadow_mask_dir', 
                        type=str, 
                        required=True, 
                        help='Directory with shadow mask data (mask files must have same name as image files)')
    parser.add_argument('--output_dir', 
                        type=str, 
                        required=True, 
                        help='Directory where to store corrected shadow images (same name as image files will be used)')
    parser.add_argument('--config_mode', 
                        type=str, 
                        required=False, 
                        help="Options: "
                             "'auto' (allocate image to nearest cluster and use prototypical configurations),"
                             "'default' (use default configuration settings),"
                             "'manual' (use configuration from custom json file)",
                        default='auto')
    parser.add_argument('--clustering_parameter_optimization_fpath', 
                        type=str, 
                        required=False, 
                        help='Path to file with clustering parameter optimization statistics',
                        default='src/shadows/shadowcorrect/optimized_cluster_parameters.json')
    parser.add_argument('--config_path', 
                        type=str, 
                        required=False, 
                        help='Path for custom configuration json file')
    parser.add_argument('--recompute_corrected_images', 
                        action='store_true', 
                        help='If flag is passed, existing shadow-corrected images will be re-processed')
    

    args = parser.parse_args()

    # read config file
    config_file = deepcopy(config)
    if args.config_path is not None:
        if Path(args.config_path).exists():
            with open(args.config_path, 'rb') as file:
                config_ = json.load(file)
            config_file.update(config_)
        else:
            raise ValueError(f'{args.config_path} does not exist')

    if args.config_mode == 'auto':
        if not Path(args.clustering_parameter_optimization_fpath).exists():
            raise ValueError("If config_mode is set to 'auto', clustering_parameter_optimization_fpath must be a valid path"
                             f"{args.clustering_parameter_optimization_fpath} not found")
        with open(args.clustering_parameter_optimization_fpath, 'rb') as file:
            config_file['clustering_parameter_optimization'] = json.load(file)

    # create directory if doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    with open(Path(args.output_dir) / 'region_group_matching_config.json', 'w') as file:
        json.dump(config_file, file)
    
    # Prepare data inputs
    image_paths = get_image_paths(args.image_dir)
    
    # Correct shadows
    correct_shadows(image_paths=image_paths, 
                    shadow_mask_dir=Path(args.shadow_mask_dir),
                    config_mode=args.config_mode, 
                    config_file=config_file, 
                    output_dir=Path(args.output_dir),
                    recompute_if_exists=args.recompute_corrected_images)
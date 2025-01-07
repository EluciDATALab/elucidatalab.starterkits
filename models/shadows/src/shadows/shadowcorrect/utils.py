from tqdm import tqdm
import numpy as np
from pyproj import Transformer

import cv2
import rasterio
import pymeanshift as pms
from sklearn.metrics import calinski_harabasz_score
from scipy.stats import skew
from scipy.ndimage import center_of_mass

import matplotlib.pyplot as plt



def get_image_and_mask(img_name, shadow_mask_dir):
    """Reads an image from file, extracts its position and acquisition time and 
    
    retrieves its shadow mask
    
    :param img_name: Pathlib. Path to image
    :param shadow_mask_dir: Path. Path to shadow mask directory
    
    :return image: 3D array. Raw image
    :return mask: 2D array. Shadow mask
    :return image_acquisition_time: str. Timestamp of acquisition, if available
    :return image_latlon: tuple. (latitude, longitude) coordinates of image center
    """
    # define mask path
    mask_path = shadow_mask_dir / f'{img_name.stem}.npy'

    if not mask_path.exists():
        raise ValueError(f'No shadow mask found for image {img_name.stem} in {shadow_mask_dir}')
        
    with rasterio.open(img_name) as src:
        R = src.read(1)
        G = src.read(2)
        B = src.read(3)

        metadata = src.meta.copy()
        
        # Retrieve geospatial metadata
        if (src.crs is not None) & (src.transform is not None):
            bounds = src.bounds  # Image bounds in projection coordinates
            crs = src.crs        # Coordinate Reference System (CRS)
            transform = src.transform  # Affine transform for pixel-to-coordinate mapping
                                
            # Calculate the center of the image in the native CRS
            center_x = (bounds.left + bounds.right) / 2
            center_y = (bounds.top + bounds.bottom) / 2
            
            # Reproject the center coordinates to latitude and longitude
            if crs != 'EPSG:4326':  # Check if the CRS is not already geographic
                transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
                lon, lat = transformer.transform(center_x, center_y)
            else:
                lon, lat = center_x, center_y  # If already in lat/lon (WGS84)
            image_latlon = (lat, lon)
        else:
            image_latlon = None
        
        # Retrieve metadata for date
        tags = src.tags()  # Retrieve metadata tags
        
        # Search for date in metadata
        image_acquisition_time = tags.get('TIFFTAG_DATETIME', None) or tags.get('DATE', None)
    image = np.dstack([R, G, B])
    
    mask = np.load(mask_path).astype(np.uint8)
    image = cv2.resize(image, (mask.shape[0], mask.shape[1]), interpolation=cv2.INTER_LINEAR)

    return image, mask, image_acquisition_time, image_latlon, metadata
    

def create_color_components_matrix(image, blob_pixels):
    """
    Creates a 3xN matrix representing the RGB color components of the specified pixels from an image, where N is the number of pixels in the blob.
    :param image: numpy.ndarray, A 3D NumPy array representing the input image with shape (H, W, 3), where: H is the height of the image. W is the width of the image. The 3rd dimension corresponds to the RGB channels.
    :param blob_pixels: tuple of numpy.ndarray, A tuple containing two 1D NumPy arrays of the same length: The first array contains the row indices (Y-coordinates) of the pixels. The second array contains the column indices (X-coordinates) of the pixels. Together, they specify the locations of the pixels to extract.
    :returns: numpy.ndarray, A 3xN NumPy array where: The first row contains the red (R) channel values of the selected pixels. The second row contains the green (G) channel values of the selected pixels. The third row contains the blue (B) channel values of the selected pixels. N is the number of pixels in `blob_pixels`.
    """
    # Extract the R, G, B values of A
    R = image[blob_pixels[0], blob_pixels[1], 0]  # Red channel
    G = image[blob_pixels[0], blob_pixels[1], 1]  # Green channel
    B = image[blob_pixels[0], blob_pixels[1], 2]  # Blue channel
    # Create a 3xN matrix where N is the number of pixels
    matrix = np.vstack([R, G, B])  # 3xN matrix
    
    return matrix

def apply_three_dimensional_space_color_transformation(matrix1, matrix2):
    """
    Applies a 3D color transformation to two input color matrices, calculating 
    their covariance matrices, performing Singular Value Decomposition (SVD), 
    and transforming the colors based on rotation, translation, and scaling.
    :param matrix1: np.ndarray, A 3-channel matrix representing the source image's color values in RGB space (each channel is an individual matrix).
    :param matrix2: np.ndarray, A 3-channel matrix representing the reference image's color values in RGB space (each channel is an individual matrix).
    :return: np.ndarray, A transformed image matrix with homogeneous coordinates applied to color transformation based on rotation, translation, and scaling matrices.
    """
    # Calculate the average for each color channel in matrix1
    avg_Rs, avg_Gs, avg_Bs = np.mean(matrix1[0]), np.mean(matrix1[1]), np.mean(matrix1[2])   
    
    # Calculate the average for each color channel in matrix2
    avg_Rr, avg_Gr, avg_Br = np.mean(matrix2[0]), np.mean(matrix2[1]), np.mean(matrix2[2])

    # Calculate the covariance matrix of matrix1 and erform Singular Value Decomposition 
    U_s, Sigma_s = calculate_covariance_and_perform_singular_value_decomposition(matrix=matrix1)
    
    # Calculate the covariance matrix of matrixR and erform Singular Value Decomposition 
    U_r, Sigma_r = calculate_covariance_and_perform_singular_value_decomposition(matrix=matrix2)
    
    # Create Rotation, Translation, Scale matrices for G_shadow and G_sunlit
    R_src, R_ref, T_src, T_ref, S_src, S_ref = create_rotation_translation_scale_matrices(U_s, 
                                                        U_r, avg_Rs, avg_Gs, avg_Bs, avg_Rr, avg_Gr, 
                                                            avg_Br, Sigma_s, Sigma_r)
    
    # Calculate the homogeneous coordinates of pixels in RGB space for G_shadow
    I_src = np.vstack([matrix1, np.ones((1, matrix1.shape[1]))]) # Stack them into a 4D matrix (R, G, B, 1)^T
    
    # Initial overall shadow removal
    I = (T_ref @ R_ref @ S_ref @ S_src @ R_src @ T_src @ I_src).T # Transpose back to Nx4 after multiplication

    I = I[:, :-1] / I[:, -1][:, np.newaxis]
    
    return I 



def calculate_covariance_and_perform_singular_value_decomposition(matrix):
    """
    Calculates the covariance matrix for the input matrix and 
    performs Singular Value Decomposition (SVD) on it.
    :param matrix: np.ndarray, A 2D numpy array representing the input data for which the covariance matrix is calculated.
    :return: A tuple containing two elements:
        - U (np.ndarray): The matrix of left singular vectors from the SVD of the covariance matrix.
        - Sigma (np.ndarray): A reordered diagonal matrix of singular values, where the reordering is based on the largest absolute values in each column of U.
    """
    # Calculate the covariance matrix of matrixS
    ##############################################
    ### Covariance Matrix:                     ###
    ### [[ var(R)      cov(R, G)   cov(R, B) ] ###
    ### [ cov(G, R)   var(G)      cov(G, B) ]  ###
    ### [ cov(B, R)   cov(B, G)   var(B)    ]] ###
    ##############################################
    cov = np.cov(matrix)
    
    # Perform Singular Value Decomposition on cov_s
    U, Sigma, _ = np.linalg.svd(cov)
    
    Sigma = np.diag(Sigma)
    
    return U, Sigma
    
def create_rotation_translation_scale_matrices(U_s, U_r, avg_Rs, avg_Gs, avg_Bs, avg_Rr, avg_Gr, avg_Br, Sigma_s, Sigma_r):
    """
    Creates rotation, translation, and scale matrices based on the singular vectors, 
    average values, and singular values of two sets of moments (source and reference).
    :param U_s: np.ndarray, Left singular vectors for the source covariance matrix (3x3).
    :param U_r: np.ndarray, Left singular vectors for the reference covariance matrix (3x3).
    :param avg_Rs: float, The average value of the red channel for the source.
    :param avg_Gs: float, The average value of the green channel for the source.
    :param avg_Bs: float, The average value of the blue channel for the source.
    :param avg_Rr: float, The average value of the red channel for the reference.
    :param avg_Gr: float, The average value of the green channel for the reference.
    :param avg_Br: float, The average value of the blue channel for the reference.
    :param Sigma_s: np.ndarray, The diagonal matrix of singular values for the source covariance matrix.
    :param Sigma_r: np.ndarray, The diagonal matrix of singular values for the reference covariance matrix.
    :return: a tuple containing six elements:
        - R_src (np.ndarray): The rotation matrix for the source image.
        - R_ref (np.ndarray): The rotation matrix for the reference image.
        - T_src (np.ndarray): The translation matrix for the source image.
        - T_ref (np.ndarray): The translation matrix for the reference image.
        - S_src (np.ndarray): The scale matrix for the source image.
        - S_ref (np.ndarray): The scale matrix for the reference image.
    """
    def fix_rotation_matrix(U):
        """
        Ensures that the input matrix U becomes a valid rotation matrix (orthogonal with det = +1).
        :param U (np.ndarray): Input matrix (3x3 or nxn).
        :return: np.ndarray, A valid rotation matrix (orthogonal with det = +1).
        """
        det = np.linalg.det(U)
        
        if np.isclose(det, 1.0):
            # Already a valid rotation matrix
            return U
        elif np.isclose(det, -1.0):
            # Fix by flipping the last column
            U[:, -1] *= -1
            return U
        else:
            # Perform QR decomposition to orthogonalize U
            Q, R = np.linalg.qr(U)
            # Check the determinant of Q after QR decomposition
            if np.linalg.det(Q) < 0:
                Q[:, -1] *= -1  # Flip last column to ensure det(Q) = +1
            return Q

    # Rotation matrix for A    
    R_src = np.eye(4)  # Create a 4x4 identity matrix
    orth_U_s = fix_rotation_matrix(U_s)
    R_src[:3, :3] = orth_U_s.transpose() # Us^-1

    # Rotation matrix for S
    R_ref = np.eye(4)
    orth_U_r = fix_rotation_matrix(U_r)
    R_ref[:3, :3] = orth_U_r # U_r
        
    # Translation matrix for A
    T_src = np.eye(4)
    T_src[0, 3] = -avg_Rs
    T_src[1, 3] = -avg_Gs
    T_src[2, 3] = -avg_Bs
    
    # Translation matrix for S
    T_ref = np.eye(4)
    T_ref[0, 3] = avg_Rr
    T_ref[1, 3] = avg_Gr
    T_ref[2, 3] = avg_Br
    
    # Scale matrix for A
    S_src = np.eye(4)
    S_src[0, 0] = np.sqrt(Sigma_s[0, 0]) ** -1 if np.sqrt(Sigma_s[0, 0]) > 0.0 else 0.0
    S_src[1, 1] = np.sqrt(Sigma_s[1, 1]) ** -1 if np.sqrt(Sigma_s[1, 1]) > 0.0 else 0.0
    S_src[2, 2] = np.sqrt(Sigma_s[2, 2]) ** -1 if np.sqrt(Sigma_s[2, 2]) > 0.0 else 0.0
    
    # Scale matrix for S
    S_ref = np.eye(4)
    S_ref[0, 0] = np.sqrt(Sigma_r[0, 0])
    S_ref[1, 1] = np.sqrt(Sigma_r[1, 1])
    S_ref[2, 2] = np.sqrt(Sigma_r[2, 2])
    
    return R_src, R_ref, T_src, T_ref, S_src, S_ref

def apply_MeanShift_clustering(image, mask, param_finetuning=None, verbose=True):
    """
    Performs Mean shift clustering on an image according to the spatial relationship of pixel point clusters and color features.
    :param image: np.ndarray, image in the LAB color space
    :param mask: np.ndarray, binary mask
    :param param_finetuning: dict, default None, dicitonary that contains the parameters needed for the MeanShift algorithm. It should contain the following keys: 'spatial_radius_values', 'range_radius_values' and 'min_density_values'.
    :param subset_size: int, default 20000. Take a subset of the pixels for clustering
    :param quantile_values: list(float), default None, quantile values for bandwidth calculation for the mean shift algorithm. if len > 1 perform grid search for optimality.
    :param verbose: boolean, default True, self explanatory.
    :return: segmented image
    """
    # Flatten the image and mask
    flat_image = image.reshape(-1, 3)
    flat_mask = mask.flatten()

    # Filter out only the unmasked pixels (where mask is non-zero)
    unmasked_indices = np.where(flat_mask == 1)[0]
    pixels_unmasked = flat_image[unmasked_indices]
    
    if param_finetuning is None:
        param_finetuning = {'spatial_radius_values': [4, 6, 8, 10], 'range_radius_values': [3.0, 4.5, 6.0, 7.5], 'min_density_values': [30, 50, 70, 90]}
        
    if 'spatial_radius_values' not in param_finetuning and 'range_radius_values' not in param_finetuning and 'min_density_values' not in param_finetuning:
        raise ValueError('Missing keys in param_finetuning. Make sure spatial_radius_values, range_radius_values and min_density_values are in the dictionary.')
    
    if ((isinstance(param_finetuning['spatial_radius_values'], int) or len(param_finetuning['spatial_radius_values']) == 1) 
        and (isinstance(param_finetuning['range_radius_values'], float) or len(param_finetuning['range_radius_values']) == 1) 
        and (isinstance(param_finetuning['min_density_values'], int) or len(param_finetuning['min_density_values']) == 1)):
        
        spatial_radius_value = param_finetuning['spatial_radius_values'] if isinstance(param_finetuning['spatial_radius_values'], int) else param_finetuning['spatial_radius_values'][0]
        range_radius_value = param_finetuning['range_radius_values'] if isinstance(param_finetuning['range_radius_values'], float) else param_finetuning['range_radius_values'][0]
        min_density_value = param_finetuning['min_density_values'] if isinstance(param_finetuning['min_density_values'], int) else param_finetuning['min_density_values'][0]
        
        # Apply Mean Shift clustering
        (_, labels_image, number_regions) = pms.segment(image, spatial_radius=spatial_radius_value, 
                                                              range_radius=range_radius_value, min_density=min_density_value)
        
        labels_image[mask == 0] = -1
        
        labels = labels_image.flatten()
        
        labels_unmasked = labels[unmasked_indices]
        
        # Calculate Calinski-Harabasz Score (only if more than 1 cluster is found)
        if number_regions > 1:
            if len(np.unique(labels_unmasked)) <= 1:
                score = -1
            else:
                score = calinski_harabasz_score(pixels_unmasked, labels_unmasked)
            if verbose:
                print(f'>>> Spatial Radius: {spatial_radius_value}, Range Radius: {round(range_radius_value, 2)}, Min Density: {min_density_value}, Calinski-Harabasz Score: {score}')
        else:
            if verbose:
                print(f'>>> Spatial Radius: {spatial_radius_value}, Range Radius: {round(range_radius_value, 2)}, Min Density: {min_density_value}, Calinski-Harabasz Score: NaN, only 1 cluster is found')
        return labels_image, {'best_spatial_radius_value':spatial_radius_value, 'best_range_radius': round(range_radius_value, 2), 'best_min_density_value': min_density_value}
    else:
        best_spatial_radius_value = None
        best_range_radius_value = None
        best_min_density_value = None
        best_calinski_harabasz_score = -1
        best_labels_image = None
        
        # Loop over the quantile values to find the best one based on Calinski-Harabasz Score
        if verbose:
            print('Peforming grid search on the quantile value for optimality...')
        for spatial_radius_value in param_finetuning['spatial_radius_values']:
            for range_radius_value in param_finetuning['range_radius_values']:
                for min_density_value in param_finetuning['min_density_values']:
                    # Apply Mean Shift clustering
                    (_, labels_image, number_regions) = pms.segment(image, spatial_radius=spatial_radius_value, 
                                                                        range_radius=range_radius_value, min_density=min_density_value)
                    
                    labels_image[mask == 0] = -1
                    labels = labels_image.flatten()
                    
                    labels_unmasked = labels[unmasked_indices]
                    
                    # Calculate Calinski-Harabasz Score (only if more than 1 cluster is found)
                    if number_regions > 1:
                        if len(np.unique(labels_unmasked)) <= 1:
                            score = -1
                        else:
                            score = calinski_harabasz_score(pixels_unmasked, labels_unmasked)
                        if verbose:
                            print(f'>>> Spatial Radius: {spatial_radius_value}, Range Radius: {round(range_radius_value, 2)}, Min Density: {min_density_value}, Calinski-Harabasz Score: {score}')
                        # Keep track of the best quantile value
                        if score > best_calinski_harabasz_score:
                            best_calinski_harabasz_score = score
                            best_spatial_radius_value = spatial_radius_value
                            best_range_radius_value = range_radius_value
                            best_min_density_value = min_density_value
                            best_labels_image = labels_image
                    else:
                        if verbose:
                            print(f'>>> Spatial Radius: {spatial_radius_value}, Range Radius: {round(range_radius_value, 2)}, Min Density: {min_density_value}, Calinski-Harabasz Score: NaN, only 1 cluster is found')
                        break
             
        if best_labels_image is None:
            # raise ValueError('No valid clustering found. All quantile values resulted in a single cluster.')
            print('No valid clustering found. All quantile values resulted in a single cluster.')
            return None,{'best_spatial_radius_value': 0, 'best_range_radius': 0.0, 'best_min_density_value': 0}
        else:
            # Use the best quantile and mean shift result
            if verbose:
                print('==========================================================================')
                print(f'>>>>>> Best Spatial Radius: {best_spatial_radius_value}, Best Range Radius: {round(best_range_radius_value, 2)}, Best Min Density: {best_min_density_value} with Calinski-Harabasz Score: {best_calinski_harabasz_score}')
                print('==========================================================================')
            return best_labels_image, {'best_spatial_radius_value': best_spatial_radius_value, 'best_range_radius': round(best_range_radius_value, 2), 'best_min_density_value': best_min_density_value}

def compute_moments(cluster_pixels):
    """
    Computes the first (mean), second (variance), and third (skewness) moments for a given cluster.
    :param cluster_pixels: np.ndarray, LAB color values for the cluster.
    :return: dict, moments for L, A, and B channels.
    """
    moments = {}

    # Compute moments for the L channel
    moments['L_mean'] = np.mean(cluster_pixels[..., 0])
    moments['L_variance'] = np.var(cluster_pixels[..., 0])
    moments['L_skewness'] = skew(cluster_pixels[..., 0].flatten())

    # Compute moments for the A channel
    moments['A_mean'] = np.mean(cluster_pixels[..., 1])
    moments['A_variance'] = np.var(cluster_pixels[..., 1])
    moments['A_skewness'] = skew(cluster_pixels[..., 1].flatten())

    # Compute moments for the B channel
    moments['B_mean'] = np.mean(cluster_pixels[..., 2])
    moments['B_variance'] = np.var(cluster_pixels[..., 2])
    moments['B_skewness'] = skew(cluster_pixels[..., 2].flatten())

    return moments

def calculate_moments_for_all_clusters(lab_image, labels):
    """
    Calculate moments for each cluster in the image.
    :param lab_image: np.ndarray, the LAB image.
    :param labels: np.ndarray, the cluster labels from Mean Shift.
    :return: dict, moments for each cluster.
    """
    unique_labels = np.unique(labels)
    cluster_moments = {}

    for label in tqdm(unique_labels):
        if label == -1:  # Masked pixels
            continue
        # Extract the pixels for the current cluster
        cluster_pixels = lab_image[labels == label]
        
        # Calculate moments for the cluster
        cluster_moments[label] = compute_moments(cluster_pixels)

    return cluster_moments

def find_global_min_max_for_moments(moment_dicts):
    """
    Finds the global minimum and maximum values for each moment (mean, variance, skewness) across all dictionaries.
    :param moment_dicts: list of dictionaries, Each dictionary contains moments (mean, variance, skewness) for L, A, and B channels.
    :return: dict, global minimum and maximum values for each moment.
    """
    # Initialize global min and max values for each moment
    global_min = {
        'L_mean': float('inf'), 'A_mean': float('inf'), 'B_mean': float('inf'),
        'L_variance': float('inf'), 'A_variance': float('inf'), 'B_variance': float('inf'),
        'L_skewness': float('inf'), 'A_skewness': float('inf'), 'B_skewness': float('inf')
    }
    global_max = {
        'L_mean': float('-inf'), 'A_mean': float('-inf'), 'B_mean': float('-inf'),
        'L_variance': float('-inf'), 'A_variance': float('-inf'), 'B_variance': float('-inf'),
        'L_skewness': float('-inf'), 'A_skewness': float('-inf'), 'B_skewness': float('-inf')
    }
    
    # Iterate through all moments dictionaries
    for moments in moment_dicts:
        for key in moments:
            global_min[key] = min(global_min[key], moments[key])
            global_max[key] = max(global_max[key], moments[key])
    
    return global_min, global_max

def prepare_clusters_for_visualization(initial_corrected_image, region_grouping, region_moments):
    """
    Prepares clusters of regions in an image for visualization by assigning colors to each 
    region and processing their moments.
    :param initial_corrected_image: np.ndarray, The original corrected image to be used as a base for visualization.
    :param region_grouping: np.ndarray, A 2D array where each pixel is assigned a region label (e.g., clusters from a segmentation algorithm).
    :param region_moments: A dictionary where each key is a region label, and each value contains the moments (e.g., mean, variance, skewness) for that region.
    :return: A tuple containing:
        - region_grouping_image (np.ndarray): A visualization of the regions, where each region is assigned a random color, and masked pixels (label == -1) are colored black.
        - processed_moments (dict): A dictionary where each region label is mapped to its assigned color and the corresponding moments.
    """
    region_grouping_image = np.zeros_like(initial_corrected_image)
    unique_region_labels = np.unique(region_grouping)
    processed_moments = {}
    for label in unique_region_labels:
        if label == -1:  # Masked pixels, assign a specific color (e.g., black)
            region_grouping_image[region_grouping == label] = [0, 0, 0]
        else:  # Assign random colors to other clusters
            mask = region_grouping == label
            color = np.random.randint(0, 255, size=(3,))
            region_grouping_image[mask] = color
            processed_moments[label] = {'color':color, 'moments': region_moments[label]}
            
    return region_grouping_image, processed_moments

def calculate_distance_ratio_blob(mask):
    """
    Calculate the distance ratio of each pixel in a blob with respect to the blob's centroid.
    :param mask: ndarray, A binary mask where pixels belonging to the blob are marked as 1, and others are 0.
    :return: ndarray, An array of the same shape as `mask`, containing distance ratios for pixels in the blob. Pixels in the blob have values representing their relative distance to the centroid, while other pixels are set to 0.
    """
    # Get the coordinates of the pixels in the blob
    blob_coords = np.column_stack(np.where(mask == 1))
    
    # Calculate the centroid (center of mass) of the blob
    centroid = center_of_mass(mask)
    
    # Calculate Euclidean distances from each pixel to the centroid
    distances = np.sqrt((blob_coords[:, 0] - centroid[0])**2 + (blob_coords[:, 1] - centroid[1])**2)
    
    # Max distance within the blob
    max_distance = distances.max()
    
    # Calculate the distance ratio for each pixel
    distance_ratio = distances / max_distance
    
    # Create an output array of the same shape as the mask
    distance_ratio_map = np.zeros_like(mask, dtype=float)
    
    # Assign the distance ratios to the corresponding positions in the blob
    distance_ratio_map[mask == 1] = distance_ratio
    
    return distance_ratio_map

def calculate_spatial_distance_weights_blob(mask, sigma_s):
    """
    Calculate spatial distance weights for pixels in a blob within a mask.
    :param mask: ndarray, A binary mask where pixels belonging to the blob are marked as 1, and others are 0.
    :param sigma_s: float, The standard deviation for the Gaussian-like spatial weight calculation, which controls the spread of the spatial weights.
    :return: A tuple containing:
        - ndarray, An array of combined spatial distance weights for each pixel in the blob, calculated by multiplying Gaussian weights with distance ratios.
        - ndarray, An array of coordinates for each pixel in the blob, where each row represents the (row, column) position of a pixel.
    """
    # Get the coordinates of the pixels in the blob
    blob_coords = np.column_stack(np.where(mask == 1))
    
    # Calculate the centroid (center of mass) of the blob
    centroid = center_of_mass(mask)
    
    # Calculate Euclidean distances from each pixel to the centroid
    distances = np.sqrt((blob_coords[:, 0] - centroid[0])**2 + (blob_coords[:, 1] - centroid[1])**2)
    
    # Apply Gaussian-like spatial weight
    gaussian_weights = np.exp(-(distances**2) / (2 * sigma_s**2))
    
    # Get distance ratio for each pixel in the blob
    distance_ratio_map = calculate_distance_ratio_blob(mask)
    
    # Retrieve the distance ratios for the blob coordinates
    distance_ratios = distance_ratio_map[mask == 1]
    
    # Calculate the combined spatial distance weights
    W_sd = gaussian_weights * distance_ratios
    
    return W_sd, blob_coords

def calculate_pixel_value_weights(image, mask, sigma_r):
    """
    Calculate the pixel value domain weight (W_pr) for each pixel in the transition zone.
    :param image: The input image (can be grayscale or color).
    :param mask: A binary mask representing the transition zone (1 for transition pixels, 0 for background).
    :param sigma_r: Standard deviation for the pixel value domain weighting.
    :return:  A tuple containing:
        - Pixel value domain weights for the transition zone.
        - ndarray, An array of coordinates for each pixel in the blob, where each row represents the (row, column) position of a pixel.
    """
    # Get the coordinates of the pixels in the transition zone
    transition_coords = np.column_stack(np.where(mask == 1))
    
    # Calculate the centroid (center pixel) in the transition zone
    center_pixel = np.mean(transition_coords, axis=0).astype(int)
    
    # Get the pixel values at the center pixel
    center_value = image[center_pixel[0], center_pixel[1]]

    # If the image is in color, calculate Euclidean distance between pixel values
    if len(image.shape) == 3:  # Color image
        center_value = center_value.astype(float)
        pixel_values = image[mask == 1].astype(float)
        value_differences = np.linalg.norm(pixel_values - center_value, axis=1)
    else:  # Grayscale image
        center_value = float(center_value)
        pixel_values = image[mask == 1].astype(float)
        value_differences = np.abs(pixel_values - center_value)
    
    # Compute pixel value domain weights (W_pr)
    W_pr = np.exp(-(value_differences ** 2) / (2 * sigma_r ** 2))
    
    return W_pr, transition_coords

def find_land_cover_type_using_MeanShift(image, param_finetuning=None, verbose=True):
    """
    Find the different types of land cover based on the MeanShift algorithm.
    :param image: RGB image after shadow correction.
    :param param_finetuning: dict, default None, dicitonary that contains the parameters needed for the MeanShift algorithm. It should contain the following keys: 'spatial_radius_values', 'range_radius_values' and 'min_density_values'.
    :param verbose: boolean, default True, self explanatory.
    :return: segmented image    
    """

    # Convert to LAB color space
    lab_image = cv2.GaussianBlur(image, (5, 5), 0)
    lab_image = cv2.cvtColor(lab_image, cv2.COLOR_RGB2LAB)

    flat_image = lab_image.reshape(-1, 3)

    if param_finetuning is None:
        param_finetuning = {'spatial_radius_values': [4, 6, 8, 10], 'range_radius_values': [3.0, 4.5, 6.0, 7.5], 'min_density_values': [30, 50, 70, 90]}
        
    if 'spatial_radius_values' not in param_finetuning and 'range_radius_values' not in param_finetuning and 'min_density_values' not in param_finetuning:
        raise ValueError('Missing keys in param_finetuning. Make sure spatial_radius_values, range_radius_values and min_density_values are in the dictionary.')

    if ((isinstance(param_finetuning['spatial_radius_values'], int) or len(param_finetuning['spatial_radius_values']) == 1) 
        and (isinstance(param_finetuning['range_radius_values'], float) or len(param_finetuning['range_radius_values']) == 1) 
        and (isinstance(param_finetuning['min_density_values'], int) or len(param_finetuning['min_density_values']) == 1)):
        
        spatial_radius_value = param_finetuning['spatial_radius_values'] if isinstance(param_finetuning['spatial_radius_values'], int) else param_finetuning['spatial_radius_values'][0]
        range_radius_value = param_finetuning['range_radius_values'] if isinstance(param_finetuning['range_radius_values'], float) else param_finetuning['range_radius_values'][0]
        min_density_value = param_finetuning['min_density_values'] if isinstance(param_finetuning['min_density_values'], int) else param_finetuning['min_density_values'][0]
        
        # Apply Mean Shift clustering
        (_, best_labels_image, number_regions) = pms.segment(image, spatial_radius=spatial_radius_value, 
                                                                range_radius=range_radius_value, min_density=min_density_value)
        
        labels = best_labels_image.flatten()
                
        # Calculate Calinski-Harabasz Score (only if more than 1 cluster is found)
        if number_regions > 1:
            score = calinski_harabasz_score(flat_image, labels)
            if verbose:
                print(f'>>> Spatial Radius: {spatial_radius_value}, Range Radius: {round(range_radius_value, 2)}, Min Density: {min_density_value}, Calinski-Harabasz Score: {score}')
        else:
            if verbose:
                print(f'>>> Spatial Radius: {spatial_radius_value}, Range Radius: {round(range_radius_value, 2)}, Min Density: {min_density_value}, Calinski-Harabasz Score: NaN, only 1 cluster is found')
    else:
        best_spatial_radius_value = None
        best_range_radius_value = None
        best_min_density_value = None
        best_calinski_harabasz_score = -1
        best_labels_image = None
        
        # Loop over the quantile values to find the best one based on Calinski-Harabasz Score
        if verbose:
            print('Peforming grid search on the quantile value for optimality...')
        for spatial_radius_value in param_finetuning['spatial_radius_values']:
            for range_radius_value in param_finetuning['range_radius_values']:
                for min_density_value in param_finetuning['min_density_values']:
                    # Apply Mean Shift clustering
                    (_, labels_image, number_regions) = pms.segment(image, spatial_radius=spatial_radius_value, 
                                                                        range_radius=range_radius_value, min_density=min_density_value)
                    
                    labels = labels_image.flatten()
                                        
                    # Calculate Calinski-Harabasz Score (only if more than 1 cluster is found)
                    if number_regions > 1:
                        score = calinski_harabasz_score(flat_image, labels)
                        if verbose:
                            print(f'>>> Spatial Radius: {spatial_radius_value}, Range Radius: {round(range_radius_value, 2)}, Min Density: {min_density_value}, Calinski-Harabasz Score: {score}')
                        # Keep track of the best quantile value
                        if score > best_calinski_harabasz_score:
                            best_calinski_harabasz_score = score
                            best_spatial_radius_value = spatial_radius_value
                            best_range_radius_value = range_radius_value
                            best_min_density_value = min_density_value
                            best_labels_image = labels_image
                    else:
                        if verbose:
                            print(f'>>> Spatial Radius: {spatial_radius_value}, Range Radius: {round(range_radius_value, 2)}, Min Density: {min_density_value}, Calinski-Harabasz Score: NaN, only 1 cluster is found')
                        break
                
        if best_labels_image is None:
            raise ValueError('No valid clustering found. All quantile values resulted in a single cluster.')
        else:
            # Use the best quantile and mean shift result
            if verbose:
                print('==========================================================================')
                print(f'>>>>>> Best Spatial Radius: {best_spatial_radius_value}, Best Range Radius: {round(best_range_radius_value, 2)}, Best Min Density: {best_min_density_value} with Calinski-Harabasz Score: {best_calinski_harabasz_score}')
                print('==========================================================================')

    if verbose:
        # Create a plot
        plt.figure(figsize=(10, 8))
        img = plt.imshow(best_labels_image, cmap='nipy_spectral')  # Use a colormap

        # Add a color bar
        cbar = plt.colorbar(img, orientation='vertical')
        cbar.set_label('Land Cover Class')  # Add a label to the color bar

        # Set the title and show the plot
        plt.title('Segmented Land Cover Types')
        plt.show()
    return best_labels_image

def convert_coordinates(x, y, source="EPSG:31370", target="EPSG:4326"):
    """
    Converts geographic coordinates from one coordinate reference system (CRS) to another.
    :param x: float or array-like, The x-coordinate(s) (longitude or easting) in the source CRS.
    :param y: float or array-like, The y-coordinate(s) (latitude or northing) in the source CRS.
    :param source: str, optional, The EPSG code of the source CRS. Default is "EPSG:31370" (Belgian Lambert 72 coordinate system).
    :param target: str, optional, The EPSG code of the target CRS. Default is "EPSG:4326" (WGS 84 geographic coordinate system, commonly used for GPS).
    :returns: tuple, Transformed coordinates (x, y) in the target CRS. If `x` and `y` are array-like, the function returns a tuple of transformed arrays.
    """
    # Define the CRS transformer (from EPSG:31370 to EPSG:4326)
    transformer = Transformer.from_crs(source, target, always_xy=True)
    
    # Transform 
    return transformer.transform(x, y)

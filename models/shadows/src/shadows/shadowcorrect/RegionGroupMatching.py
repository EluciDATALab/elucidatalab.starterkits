from tqdm import tqdm
import numpy as np
from pvlib import solarposition

import cv2
from skimage import color
from skimage.feature import local_binary_pattern
from scipy.spatial import cKDTree

import shadows.shadowcorrect.utils as rgm_utils


class RegionGroupMatching():
    def __init__(self, original_image, shadow_mask, 
                 image_latlon=None, image_acquisition_time=None, sun_orientation_exclusion_range=90):
        """
        Performs the shadow correction algorithm mentioned in the following paper: 
        Guo, Mingqiang, et al. "Shadow removal method for high-resolution aerial 
        remote sensing images based on region group matching." Expert Systems with 
        Applications 255 (2024): 124739. (https://www.sciencedirect.com/science/article/abs/pii/S0957417424016063)
        :param original_image: numpy array, the original image that needs shadow compensation. 
        :param shadow_mask: numpy array, the shadow (binary) mask of the image
        :param image_latlon: tuple. Latitude and longitude coordinates of image. Default=None
        :para image_acquisition_time: str. Time of image acquisition. Default=None
        :param sun_orientation_exclusion_range: float. If image_acquisition_time and latlon are provided,
        this range, centered on the sun's direction, will be excluded from the partially light boundary
        around a shadow blob. Default=90
        """
        self.original_image = original_image
        self.shadow_mask = shadow_mask
        self.sun_orientation = None
        self.exclude_angles = None
        if (image_latlon is not None) & (image_acquisition_time is not None):
            self.sun_orientation = solarposition.get_solarposition(
                image_acquisition_time, *image_latlon)['azimuth'].values[0]
            self.exclude_angles = (self.sun_orientation - sun_orientation_exclusion_range / 2,
                                   self.sun_orientation + sun_orientation_exclusion_range / 2)
    
    # -------------- STEP 1 --------------
    def initial_overall_shadow_removal(self, kernel_sigma=10, exposure_factor=4):
        """
         Performs the first step of the proposed method. Based on the shadow region mask of 
         the aerial remote sensing image, the image is divided into two parts: the shadow 
         region and the light region. Crop the light region using an expansion region to 
         obtain a portion of the light region around the shadow region. Utilizing color 
         transformation based on 3D space, a partial light region is used to perform 
         overall color transformation on the shadow region. 
         :param kernel_sigma: int, default value 10, value of the kernel sigma for mask dilation
         :param exposure_factor: float. Change exposure of shadow areas by this factor for initial
         correction. Set 1 for no exposure correction. Default=4
        """
        # Check if the mask contains only 0 and 1
        if not np.array_equal(np.unique(self.shadow_mask), np.array([0, 1])):
            self.shadow_mask = self.shadow_mask / 255.0
            
        # Find connected components in the mask (blobs)
        num_labels, labels_im = cv2.connectedComponents(self.shadow_mask)

        self.initial_corrected_image = self.original_image.copy()

        self.normalized_image = self.original_image.copy() / 255.0
        self.normalized_image_exposure_adjusted = adjust_exposure(self.original_image.copy(), exposure_factor) / 255.0

        # Iterate through each blob (label) and process them
        for label in range(1, num_labels):  # Start from 1, as 0 is the background
            # Create a mask for the current blob
            blob_mask = (labels_im == label).astype(np.uint8)
            if blob_mask.sum() < 50:
                continue

            # Extract the partially light region
            dilated_blob_mask = cv2.dilate(blob_mask, np.ones((kernel_sigma, kernel_sigma), np.uint8), iterations=1)
            extra_dilated_blob_mask = cv2.subtract(dilated_blob_mask, blob_mask)

            if (self.exclude_angles is not None) & (blob_mask.sum() > 100):
                extra_dilated_blob_mask = filter_contour_by_angle(blob_mask, extra_dilated_blob_mask, angle_range=self.exclude_angles)
            
            # Extract RGB values of A
            blob_pixels = np.where(blob_mask == 1)
            
            # Create a 3xN matrix where N is the number of pixels in A
            matrixS = rgm_utils.create_color_components_matrix(image=self.normalized_image_exposure_adjusted, blob_pixels=blob_pixels)
            
            # Extract RGB values of S
            dilated_blob_pixels = np.where(extra_dilated_blob_mask == 1)
            
            # Create a 3xM matrix where M is the number of pixels in S
            matrixR = rgm_utils.create_color_components_matrix(image=self.normalized_image, blob_pixels=dilated_blob_pixels)
            
            # Initial overall shadow removal
            I = rgm_utils.apply_three_dimensional_space_color_transformation(matrix1=matrixS, matrix2=matrixR)

            I = (I * 255).astype(np.uint8) # Because the image is normalized now
            
            # After processing, reconstruct the shadow blob back to its original shape
            # Assign the modified R, G, B values back to the original image positions
            new_R, new_G, new_B = I[:, 0], I[:, 1], I[:, 2] 
            
            # # Reassign the modified pixel values back to the original positions in the image
            self.initial_corrected_image[blob_pixels[0], blob_pixels[1], 0] = np.uint8(new_R)  
            self.initial_corrected_image[blob_pixels[0], blob_pixels[1], 1] = np.uint8(new_G) 
            self.initial_corrected_image[blob_pixels[0], blob_pixels[1], 2] = np.uint8(new_B) 

    # -------------- STEP 2 --------------
    def internal_grouping_of_shadow_and_light_regions(self, shadow_finetuning=None, light_finetuning=None, verbose=True):
        """
        Based on the spatial and color features of remote sensing images, the preliminary results of 
        overall shadow removal in the shadow region and the segmentation of the light region into 
        multiple irregular small regions are divided, respectively. Then, the color moments 
        principle roughly groups the preliminary results of overall shadow removal and the irregular 
        small region inside the light region, obtaining several shadow and light groups.
        :param shadow_finetuning: dict, default None. Dictionary that contains the parameters needed for the MeanShift algorithm. It should contain the following keys: 'spatial_radius_values', 'range_radius_values' and 'min_density_values'.
        :param light_finetuning: dict, default None. Dictionary that contains the parameters needed for the MeanShift algorithm. It should contain the following keys: 'spatial_radius_values', 'range_radius_values' and 'min_density_values'.
        :param verbose: boolean, default True, self explanatory.
        """
        # Convert to LAB color space
        self.lab_image = cv2.GaussianBlur(self.initial_corrected_image, (5, 5), 0)
        self.lab_image = cv2.cvtColor(self.lab_image, cv2.COLOR_RGB2LAB)

        # Create an empty copy of the image for each region
        shadow_regions = np.zeros_like(self.lab_image)  # Image where mask == 1
        light_regions = np.zeros_like(self.lab_image)  # Image where mask == 0

        # Apply the mask to create the two images
        shadow_regions[self.shadow_mask == 1] = self.lab_image[self.shadow_mask == 1]  # Keep pixels where mask is 1
        light_regions[self.shadow_mask == 0] = self.lab_image[self.shadow_mask == 0]  # Keep pixels where mask is 0
                
        # Group shadow regions
        if verbose:
            print('Applying the MeanShift algorithm on the shadow region...')
        self.shadow_region_grouping, _ = rgm_utils.apply_MeanShift_clustering(image=shadow_regions, mask=self.shadow_mask, param_finetuning=shadow_finetuning, verbose=verbose)
                
        # Group light regions
        if verbose:
            print('Applying the MeanShift algorithm on the light region...')
        self.light_region_grouping, _ = rgm_utils.apply_MeanShift_clustering(image=light_regions, mask=(1 - self.shadow_mask), param_finetuning=light_finetuning, verbose=verbose)
        
        if verbose:
            print('Calculating moments (mean, variance and skewness) per component for each cluster in both regions...')
        # Calculate moments for each cluster in the shadow region
        self.shadow_region_moments = rgm_utils.calculate_moments_for_all_clusters(lab_image=self.lab_image, labels=self.shadow_region_grouping)
        
        # Calculate moments for each cluster in the light region
        self.light_region_moments = rgm_utils.calculate_moments_for_all_clusters(lab_image=self.lab_image, labels=self.light_region_grouping)
        
    # -------------- STEP 3 --------------
    def feature_extraction_and_matching(self, radius=2, n_points=16, gradient_kernel_size=3):
        """
        According to the grouping of shadow and lighting groups, extract local texture and color features of 
        nine color channels and two gradient directions for a total of 11 channels in each group of images, 
        and combine the average texture feature values of the 11 channels to construct texture feature vectors 
        for expressing each group of images. Based on this, match between shadow and light groups.
        :param radius: int, default 2, defines the distance between the center pixel and the sampling points around it.
        :param n_points: int, default 16, determines how many points are taken in the circular neighborhood around the center pixel.
        :param gradient_kernel_size: int, default 3, kernel size for Sobel filtering.
        """
        def euclidean_distance(vec1, vec2):
            return np.linalg.norm(np.array(vec1) - np.array(vec2))
        
        def min_max_normalize_channel(values, min_val, max_val):
            return (values - min_val) / (max_val - min_val)
        
        self.lab_image = self.initial_corrected_image / 255.0
        self.lab_image = color.rgb2lab(self.lab_image).astype(int)
        
        # Convert to HSV color space
        self.hsv_image = self.initial_corrected_image / 255.0
        self.hsv_image = color.rgb2hsv(self.hsv_image)
        self.hsv_image[..., 0] = self.hsv_image[..., 0] * 360 # Multiply the H channel by 360 to convert it to [0, 360] range
        self.hsv_image[..., 1] = self.hsv_image[..., 1] * 100 # Multiply the S channel by 100 to convert it to [0, 100] range
        self.hsv_image[..., 2] = self.hsv_image[..., 2] * 100 # Multiply the V channel by 100 to convert it to [0, 100] range
        self.hsv_image = self.hsv_image.astype(int)
        
        # retrieve mean eigenvectors
        self.shadow_mean_eigenvectors = self.__calculate_local_texture_feature_matrices_and_retrieve_mean_eigenvectors(region_grouping=self.shadow_region_grouping, radius=radius, n_points=n_points, gradient_kernel_size=gradient_kernel_size)
        self.light_mean_eigenvectors = self.__calculate_local_texture_feature_matrices_and_retrieve_mean_eigenvectors(region_grouping=self.light_region_grouping, radius=radius, n_points=n_points, gradient_kernel_size=gradient_kernel_size)
        
        # Initialize min and max values for each channel across all shadow and light regions
        all_channels = list(self.shadow_mean_eigenvectors.values())[0].keys()
        channel_min_max = {channel: {'min': float('inf'), 'max': float('-inf')} for channel in all_channels}

        # Find global min and max for each channel
        for eigenvectors in [self.shadow_mean_eigenvectors, self.light_mean_eigenvectors]:
            for values in eigenvectors.values():
                for channel, value in values.items():
                    if value < channel_min_max[channel]['min']:
                        channel_min_max[channel]['min'] = value
                    if value > channel_min_max[channel]['max']:
                        channel_min_max[channel]['max'] = value
        
        # Calculate Euclidean distance between the mean eigenvectors of shadow and light regions
        self.closest_labels = {}
        for shadow_label, shadow_values in self.shadow_mean_eigenvectors.items():
            # Normalize shadow eigenvector per channel
            shadow_eigenvector = np.array([min_max_normalize_channel(shadow_values[channel], 
                                                                    channel_min_max[channel]['min'], 
                                                                    channel_min_max[channel]['max']) 
                                        for channel in all_channels])
                        
            min_distance = float('inf')
            closest_label = None
            for light_label, light_values in self.light_mean_eigenvectors.items():
                # Normalize light eigenvector per channel
                light_eigenvector = np.array([min_max_normalize_channel(light_values[channel], 
                                                                        channel_min_max[channel]['min'], 
                                                                        channel_min_max[channel]['max']) 
                                            for channel in all_channels])
                
                distance = euclidean_distance(shadow_eigenvector, light_eigenvector)
                                
                if distance < min_distance:
                    min_distance = distance
                    closest_label = light_label
                    
            self.closest_labels[shadow_label] = closest_label

    def __calculate_local_texture_feature_matrices_and_retrieve_mean_eigenvectors(self, region_grouping, radius, n_points, gradient_kernel_size):
        """
        Calculates the local texture feature matrices using LBP (Local Binary pattern) 
        for the channels R, G, B, H, S, V, L, A, B and the X, Y gradient directions. 
        :param region_grouping: np.array, segmented image
        :param radius: int, defines the distance between the center pixel and the sampling points around it.
        :param n_points: int, determines how many points are taken in the circular neighborhood around the center pixel.
        :param gradient_kernel_size: int, kernel size for Sobel filtering
        :return: dict of mean local_texture_feature_matrices values per cluster
        """
        FEATURE_MATRICES_KEYS = ['R', 'G', 'Bl', 'H', 'S', 'V', 'L', 'A', 'B']
        mean_eigenvectors = {}
        
        unique_labels = np.unique(region_grouping)
        # Loop over different clusters
        for label in tqdm(unique_labels):
            if label == -1:  # Masked pixels
                continue
            
            tmp_mean_eigenvector = {}
            # Loop over images
            ix = 0
            for image in [self.initial_corrected_image, self.hsv_image, self.lab_image]:                        
                # Expand the 2D mask (region_grouping == label) to 3D by repeating it along the channel axis
                mask_3d = np.repeat((region_grouping == label)[:, :, np.newaxis], 3, axis=2)
        
                # Apply the expanded mask to the image (this preserves the spatial dimensions)
                cluster_pixels = np.where(mask_3d, image, 0)  # Set the pixels outside the region to 0
                
                # Split channels
                channels = cv2.split(cluster_pixels)
                for channel in channels:
                    lbp = local_binary_pattern(channel, P=n_points, R=radius, method='ror')
                    tmp_mean_eigenvector[FEATURE_MATRICES_KEYS[ix]] = np.mean(lbp[region_grouping == label])  # Mean value of LBP matrix
                    ix+=1
            
            # grayscale the image
            gray_image = cv2.cvtColor(self.initial_corrected_image, cv2.COLOR_RGB2GRAY)
            
            gray_cluster_pixels = np.zeros_like(gray_image)
            gray_cluster_pixels[region_grouping == label] = gray_image[region_grouping == label]
            
            # Compute the X gradient (derivative in the X direction)
            gradient_x = cv2.Sobel(gray_cluster_pixels, cv2.CV_64F, 1, 0, ksize=gradient_kernel_size).astype(int)
            lbp_x = local_binary_pattern(gradient_x, P=n_points, R=radius, method='ror')
            tmp_mean_eigenvector['X'] = np.mean(lbp_x[region_grouping == label])  # Mean value of LBP matrix

            # Compute the Y gradient (derivative in the Y direction)
            gradient_y = cv2.Sobel(gray_cluster_pixels, cv2.CV_64F, 0, 1, ksize=gradient_kernel_size).astype(int)
            lbp_y = local_binary_pattern(gradient_y, P=n_points, R=radius, method='ror')
            tmp_mean_eigenvector['Y'] = np.mean(lbp_y[region_grouping == label])  # Mean value of LBP matrix
            
            mean_eigenvectors[label] = tmp_mean_eigenvector
            
        return mean_eigenvectors
            
    # -------------- STEP 4 --------------
    def local_shadow_region_enhancement(self, threshold=0.2, verbose=True):
        """
        After the matching between shadow and light groups is completed, the matching results are 
        corrected according to each group of shadows and light color features. Then, for each 
        group of matched shadow and light blocks, the shadow removal effect is again enhanced 
        using the three-dimensional space color transformation. Finally, concatenate all 
        restored shadow groups to obtain the final image shadow removal result.
        :param threshold: float, default 0.2, makes sure the shadow and sunlit regions have similar moments.
        :param verbose: boolean, default True, self explanatory. 
        """
        def normalize_moments(moments, min_stats, max_stats):
            """
            Min-Max normalization for the input moments.
            :param moments: dict, moments for a single region.
            :param min_stats: dict, minimum values of each moment from the dataset.
            :param max_stats: dict, maximum values of each moment from the dataset.
            :return: np.ndarray, normalized moments as a 1D array.
            """
            normalized = []
            
            for key in moments:
                # Min-Max normalization
                normalized_value = (moments[key] - min_stats[key]) / (max_stats[key] - min_stats[key])
                normalized.append(normalized_value)
            
            return np.array(normalized)
        
        self.enhanced_corrected_image = self.initial_corrected_image.copy()

        self.normalized_initial_corrected_image = self.initial_corrected_image.copy() / 255.0
        
        # Prepare moments for global min/max
        shadow_moments_values = np.array([val for val in self.shadow_region_moments.values()]).flatten()
        light_moments_values = np.array([val for val in self.light_region_moments.values()]).flatten()
        combined_moments = np.concatenate([shadow_moments_values, light_moments_values])
        
        global_min, global_max = rgm_utils.find_global_min_max_for_moments(moment_dicts=combined_moments)
        
        for shadow_label, light_label in tqdm(self.closest_labels.items()):
            shadow_moments = self.shadow_region_moments[shadow_label]
            light_moments = self.light_region_moments[light_label]
            
            # Normalize the moments
            shadow_moments_normalized = normalize_moments(shadow_moments, global_min, global_max)
            light_moments_normalized = normalize_moments(light_moments, global_min, global_max)

            # Compute the Euclidean distance with normalized moments
            distance = np.linalg.norm(shadow_moments_normalized - light_moments_normalized)
            
            if distance < threshold:
                if verbose:
                    print(f">>> Shadow region {shadow_label} and light region {light_label} are similar (distance: {distance:.4f} < threshold: {threshold:.4f})")
                
                # Apply the expanded mask to the image (this preserves the spatial dimensions)
                G_shadow = np.where(self.shadow_region_grouping == shadow_label) 
            
                # Apply the expanded mask to the image (this preserves the spatial dimensions)
                G_sunlit = np.where(self.light_region_grouping == light_label) 
                
                # Create a 3xN matrix where N is the number of pixels in G_shadow
                matrixG_shadow = rgm_utils.create_color_components_matrix(image=self.normalized_initial_corrected_image, blob_pixels=G_shadow)
                # matrixG_shadow = rgm_utils.create_color_components_matrix(image=self.initial_corrected_image, blob_pixels=G_shadow)
                
                # Create a 3xM matrix where M is the number of pixels in G_sunlit
                matrixG_sunlit = rgm_utils.create_color_components_matrix(image=self.normalized_initial_corrected_image, blob_pixels=G_sunlit)
                # matrixG_sunlit = rgm_utils.create_color_components_matrix(image=self.initial_corrected_image, blob_pixels=G_sunlit)
                
                # Overall shadow removal
                I = rgm_utils.apply_three_dimensional_space_color_transformation(matrix1=matrixG_shadow, matrix2=matrixG_sunlit)

                I = (I * 255).astype(np.uint8) # Because the image is normalized now
                
                # After processing, reconstruct the shadow blob back to its original shape
                # Assign the modified R, G, B values back to the original image positions
                new_R, new_G, new_B = I[:, 0], I[:, 1], I[:, 2] 

                # # Reassign the modified pixel values back to the original positions in the image
                self.enhanced_corrected_image[G_shadow[0], G_shadow[1], 0] = np.uint8(new_R)  
                self.enhanced_corrected_image[G_shadow[0], G_shadow[1], 1] = np.uint8(new_G) 
                self.enhanced_corrected_image[G_shadow[0], G_shadow[1], 2] = np.uint8(new_B) 
            else:
                if verbose:
                    print(f" >>> WARNING: Shadow region {shadow_label} and light region {light_label} are not similar (distance: {distance:.4f} >= threshold: {threshold:.4f}) <<<")


    # -------------- STEP 5 --------------   
    def shadow_boundary_optimization(self, kernel_size=3, spatial_weight=1.0, pixel_value_weight=10.0, lambda_factor=0.5):
        """
        The final shadow removal result still shows boundary effects in the shadow boundary section. 
        Dynamically weighted boundary optimization is performed on the shadow edge by combining 
        spatial and range information between pixels. 
        :param kernel_size: int, default 3, Size of the neighborhood (3x3, 5x5, etc.). 
        :param spatial_weight: float, default 1.0, controls the influence of spatial distance between pixels.
        :param pixel_value_weights: float, default 10.0, controls the influence of pixel value differences (color or intensity).
        :param lambda_factor: float, default 0.5, scalar weight factor to balance the weights.
        """
        def ensure_odd_kernel_size(kernel_size):
            # If kernel_size is even, make it odd by adding 1
            if kernel_size % 2 == 0:
                kernel_size += 1
            return kernel_size
        
        kernel_size = ensure_odd_kernel_size(kernel_size)
        
        self.final_corrected_image = self.enhanced_corrected_image.copy()

        # Find connected components in the mask (blobs)
        num_labels, labels_im = cv2.connectedComponents(self.shadow_mask)

        # Iterate through each blob (label) and process them
        for shadow_label in tqdm(range(1, num_labels)):  # Start from 1, as 0 is the background
            # Create a mask for the current blob
            shadow_mask = (labels_im == shadow_label).astype(np.uint8)

            # Dilate the shadow region to include some of the neighboring light pixels
            kernel = np.ones((kernel_size, kernel_size), np.uint8) 
            dilated_mask = cv2.dilate(shadow_mask, kernel, iterations=1)

            # Define how much of the shadow region and light region you want to include in the transition zone.
            eroded_mask = cv2.erode(shadow_mask, kernel, iterations=1)
            
            # Combine the shadow edge and the light region using logical operations
            shadow_edge = np.logical_and(shadow_mask == 1, eroded_mask == 0)
            light_region = np.logical_and(dilated_mask == 1, shadow_mask == 0)

            # Combine the shadow edge and light region to form the transition zone
            transition_zone = np.logical_or(shadow_edge, light_region)

            # Convert the transition zone to a binary mask (0 or 1)
            transition_zone_mask = transition_zone.astype(np.uint8)

            blob_coords = np.column_stack(np.where(shadow_mask == 1))
            
            # # Apply weighted smoothing to adjust the pixel values in the transition zone
            smoothed_image = self.__apply_weighted_smoothing(self.enhanced_corrected_image, blob_coords, sigma_s=spatial_weight, sigma_r=pixel_value_weight, lambda_factor=lambda_factor)
            
            # Update only the pixels in the transition zone
            self.final_corrected_image[transition_zone_mask == 1] = smoothed_image[transition_zone_mask == 1]

    def __apply_weighted_smoothing(self, image, blob_coords, sigma_s, sigma_r, lambda_factor):
        """
        Apply weighted smoothing to calculate the new pixel 
        value at the center of the transition zone  based on Equation 25.
        :param image: The input image (or the region of interest where smoothing is applied).
        :param blob_coords: list, of X, Y coordinates
        :param W: Combined weight from Equation 24.
        :param kernel_size: int, kenrel size for the neighborhood.
        :return:  The smoothed image where transition zones are adjusted.
        """
        
        # Initialize the smoothed image
        smoothed_image = np.copy(image)
        
        # blob_coords_xy = blob_coords[:, [1, 0]]  # Swap y and x
        
        # for (x_center, y_center) in blob_coords:
        for (y_center, x_center) in blob_coords:
            # Get the neighborhood around the center pixel
            neighborhood = self.__get_neighborhood(image, (y_center, x_center), kernel_size=5)

            # Create a binary mask for the neighborhood
            neighborhood_mask = np.ones_like(neighborhood[..., 0], dtype=np.uint8)  # Assuming 2D mask

            # Calculate spatial distance weights (W_sd) for the neighborhood
            W_sd, _ = rgm_utils.calculate_spatial_distance_weights_blob(mask=neighborhood_mask, sigma_s=sigma_s)
                        
            # Calculate pixel value weights (W_pr) for the neighborhood
            W_pr, _ = rgm_utils.calculate_pixel_value_weights(image=neighborhood, mask=neighborhood_mask, sigma_r=sigma_r)
            
            # Combine the weights
            W = W_sd * W_pr * lambda_factor
            
            # Reshape weights to match the spatial dimensions of the neighborhood
            weights = W.reshape(5, 5, 1)  # Now (5, 5, 1)

            # Broadcast weights to match the neighborhood's channel dimension
            weights = np.repeat(weights, neighborhood.shape[-1], axis=-1)  # Now (5, 5, 3)

            # Calculate weighted sum of pixel values (numerator in Equation 25)
            weighted_sum = np.sum(neighborhood * weights)  # Sum over spatial dimensions
            
            # Normalize by the sum of weights (denominator in Equation 25)
            sum_of_weights = np.sum(weights)

            # Update the center pixel value
            smoothed_image[x_center, y_center] = (weighted_sum / sum_of_weights).astype(np.uint8)
        
        return smoothed_image

    def __get_neighborhood(self, array, center, kernel_size):
        """
        Extract a neighborhood of pixels around a center pixel in an array.        
        :param array: The input array (image or weight matrix).
        :param center: The (x, y) coordinates of the center pixel.
        :param kernel_size: Size of the neighborhood (3x3, 5x5, etc.).        
        :return: The extracted neighborhood as a subarray.
        """
        x_center, y_center = center
        half_size = kernel_size // 2
        
        # Pad the array to handle edge cases where the neighborhood extends beyond the array boundary
        # Check the shape of the array to apply the correct padding
        if array.ndim == 2:  # If the array is 2D (e.g., grayscale or weight matrix)
            padded_array = np.pad(array, ((half_size, half_size), (half_size, half_size)), mode='constant', constant_values=0)
        elif array.ndim == 3:  # If the array is 3D (e.g., color image with channels)
            padded_array = np.pad(array, ((half_size, half_size), (half_size, half_size), (0, 0)), mode='constant', constant_values=0)
        else:
            raise ValueError("Unsupported array shape for neighborhood extraction. Array must be 2D or 3D.")
        
        # Adjust coordinates due to padding
        x_center_padded = x_center + half_size
        y_center_padded = y_center + half_size
        
        # Extract the neighborhood window from the padded array
        neighborhood = padded_array[x_center_padded - half_size:x_center_padded + half_size + 1, 
                                    y_center_padded - half_size:y_center_padded + half_size + 1]
        
        return neighborhood


def filter_contour_by_angle(mask, contour, angle_range):
    """
    Remove contour pixels at specific angles with respect to the nearest shadow mask pixel.

    Args:
        mask (2D array): Binary shadow mask array.
        contour (2D array): Binary contour array.
        angle_range (tuple): Range of angles to remove (min_angle, max_angle) in degrees.

    Returns:
        filtered_contour (2D array): Contour array with filtered pixels.
    """
    # Get coordinates of shadow mask pixels
    shadow_coords = np.argwhere(mask == 1)

    # Get coordinates of contour pixels
    contour_coords = np.argwhere(contour == 1)

    # Build a KDTree for efficient nearest-neighbor search
    tree = cKDTree(shadow_coords)

    # Find the nearest shadow pixel for each contour pixel
    distances, nearest_indices = tree.query(contour_coords, k=10)

    # Get the nearest shadow pixel coordinates for each contour pixel
    nearest_shadow_coords = np.stack([np.median(shadow_coords[[n0 for n0 in n if n0 < (shadow_coords.shape[0])]], axis=0)
                                      for n in nearest_indices]) #shadow_coords[nearest_indices]

    # Compute the vector differences
    vectors = nearest_shadow_coords - contour_coords  # Shape: (n, 2)

    # Calculate angles using atan2 (y is negated for North=0)
    angles_rad = np.arctan2(-vectors[:, 1], vectors[:, 0])  # Radians

    # Convert to degrees and adjust to [0, 360)
    angles = (np.degrees(angles_rad) + 360) % 360

    # Create a mask for angles outside the specified range
    min_angle, max_angle = angle_range
    if min_angle < max_angle:
        valid_indices = (angles < min_angle) | (angles > max_angle)
    else:  # Handle angle ranges that wrap around 0 (e.g., 350 to 10 degrees)
        valid_indices = (angles > max_angle) & (angles < min_angle)

    # Filter contour coordinates
    filtered_coords = contour_coords[valid_indices]

    # Create the filtered contour array
    filtered_contour = np.zeros_like(contour)
    filtered_contour[filtered_coords[:, 0], filtered_coords[:, 1]] = 1

    return filtered_contour


def adjust_exposure(image, exposure_factor):
    """
    Adjusts the exposure of an input image by scaling its pixel values.
    :param image: numpy.ndarray, A 2D or 3D NumPy array representing the input image. The image should be in 8-bit format with pixel values in the range [0, 255]. It can be a grayscale image (single channel) or an RGB image (three channels).
    :param exposure_factor : float, A scaling factor for adjusting exposure. Values > 1 will increase exposure (brighten the image). Values < 1 will decrease exposure (darken the image). Values must be positive.
    :returns: numpy.ndarray, A NumPy array of the same shape as the input image, with adjusted pixel values. The output image is in 8-bit format with pixel values in the range [0, 255].
    """
    # Convert to float to avoid clipping during adjustment
    image_float = image.astype(np.float32) / 255.0
    
    # Adjust exposure: scale pixel values (increase for brighter exposure)
    adjusted_image = np.clip(image_float * exposure_factor, 0, 1)
    
    # Convert back to 8-bit
    adjusted_image = (adjusted_image * 255).astype(np.uint8)
    return adjusted_image
    
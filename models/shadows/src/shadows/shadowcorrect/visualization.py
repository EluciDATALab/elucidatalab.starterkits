import matplotlib.pyplot as plt
import cv2
import numpy as np
import random

def display_image_and_mask(image, mask, image_name, figsize=(10,5)):
    """
    Displays an image and its corresponding mask side by side using matplotlib.
    :param image: np.ndarray, The input image to be displayed.
    :param mask: np.ndarray, The binary mask corresponding to the image, which will be displayed in grayscale.
    :param image_name: str, A string representing the name or identifier of the image, which will be used in the titles.
    :param figsize: tuple(int), default (10,5), A tuple specifying the size of the figure.
    """
    _, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].imshow(image)
    axes[0].set_title(f'Image: {image_name}')
    axes[0].axis('off')

    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title(f'Mask: {image_name}')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()
    
def display_shadow_and_partially_light_regions(image, mask, kernel_sigma, figsize=(10, 5)):
    """
    Displays the shadow regions, partially light regions with shadows, 
    and purely partially light regions of an image side by side.
    :param image: np.ndarray, The input image where shadows and partially light regions are to be visualized.
    :param mask: np.ndarray, A binary mask that highlights shadow regions in the image.
    :param kernel_sigma: int, The kernel size for dilation of the mask to find the partially light regions.
    :param figsize: tuple(int), default (10,5), A tuple specifying the size of the figure.
    """
    ### Retrieve shadow regions; Each shadow region will be denoted as A
    shadow_regions = cv2.bitwise_and(image, image, mask=mask)

    ### Retrieve the partially light regions; Each partial light region will be denoted as S
    dilated_mask = cv2.dilate(mask, np.ones((kernel_sigma, kernel_sigma), np.uint8), iterations=1)
    partially_light_regions_with_shadows = cv2.bitwise_and(image, image, mask=dilated_mask)

    # Find the difference between the dilated mask and the original mask
    extra_dilated_mask = cv2.subtract(dilated_mask, mask)
    partially_light_regions = cv2.bitwise_and(image, image, mask=extra_dilated_mask)

    _, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].imshow(shadow_regions)
    axes[0].set_title(f'Shadow regions')
    axes[0].axis('off')

    axes[1].imshow(partially_light_regions_with_shadows)
    axes[1].set_title(f'Partially light regions with shadows')
    axes[1].axis('off')

    axes[2].imshow(partially_light_regions)
    axes[2].set_title(f'Partially light regions')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()
    
def display_initial_overall_shadow_removal_in_shadow_regions(image, corrected_image, image_name, figsize=(10, 5)):
    """
    Displays the initial shadow regions and the corresponding corrected image after shadow removal side by side.
    :param image: np.ndarray, The original image with shadow regions.
    :param corrected_image: np.ndarray, The corrected image where shadows have been removed.
    :param image_name: str, A string representing the name or identifier of the image, which will be used in the titles.
    :param figsize: tuple(int), default (10,5), A tuple specifying the size of the figure.
    """
    _, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].imshow(image)
    axes[0].set_title(f'Image: {image_name}')
    axes[0].axis('off')

    axes[1].imshow(corrected_image)
    axes[1].set_title(f'Initial corrected image: {image_name}')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()
    
def display_internal_grouping_of_image_shadow_and_light_regions(shadow_region_grouping_image, light_region_grouping_image, image_name, figsize=(10, 5)):
    """
    Displays the internal grouping of shadow regions and light regions side by side.
    :param shadow_region_grouping_image: np.ndarray, An image where different shadow regions are grouped and visualized with distinct colors.
    :param light_region_grouping_image: np.ndarray, An image where different light regions are grouped and visualized with distinct colors.
    :param image_name: str, A string representing the name or identifier of the image, which will be used in the titles.
    :param figsize: tuple(int), default (10,5), A tuple specifying the size of the figure.
    """
    _, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].imshow(shadow_region_grouping_image)
    axes[0].set_title(f'Shadow region grouping: {image_name}')
    axes[0].axis('off')

    axes[1].imshow(light_region_grouping_image)
    axes[1].set_title(f'Light region grouping: {image_name}')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()
    
def scatter_plot_moments(processed_moments_1, processed_moments_2):
    """
    Creates a scatter plot for the moments (mean, variance, and skewness) 
    of shadow and light regions, using different colors for each region.
    :param processed_moments_1: dict, A dictionary containing the moments (mean, variance, skewness) and colors for shadow regions.
    :param processed_moments_2: dict, A dictionary containing the moments (mean, variance, skewness) and colors for light regions.
    """
    def normalize_color(rgb_array):
        """
        Normalizes an RGB color array by dividing each element by 255, 
        converting the values from the range [0, 255] to the range [0, 1].
        :param rgb_array: np.ndarray or list, A 1D or 2D array or list representing the RGB color values. Each value should be in the range [0, 255].
        :return: np.ndarray, A numpy array with the normalized RGB values, where each value is scaled to the range [0, 1].
        """
        return rgb_array / 255.0

    # Assuming processed_moments_1 and processed_moments_2 are your two dictionaries
    labels_dict = {
        "Shadow Moments": list(processed_moments_1.keys()),
        "Light Moments": list(processed_moments_2.keys())
    }

    colors_dict = {
        "Shadow Moments": [normalize_color(processed_moments_1[label]['color']) for label in labels_dict["Shadow Moments"]],
        "Light Moments": [normalize_color(processed_moments_2[label]['color']) for label in labels_dict["Light Moments"]]
    }

    # Moments dictionary extraction for both dictionaries
    moments_dict = {
        "Shadow Moments": {
            'mean': [[processed_moments_1[label]['moments'][f'{ch}_mean'] for label in labels_dict["Shadow Moments"]] for ch in ['L', 'A', 'B']],
            'variance': [[processed_moments_1[label]['moments'][f'{ch}_variance'] for label in labels_dict["Shadow Moments"]] for ch in ['L', 'A', 'B']],
            'skewness': [[processed_moments_1[label]['moments'][f'{ch}_skewness'] for label in labels_dict["Shadow Moments"]] for ch in ['L', 'A', 'B']]
        },
        "Light Moments": {
            'mean': [[processed_moments_2[label]['moments'][f'{ch}_mean'] for label in labels_dict["Light Moments"]] for ch in ['L', 'A', 'B']],
            'variance': [[processed_moments_2[label]['moments'][f'{ch}_variance'] for label in labels_dict["Light Moments"]] for ch in ['L', 'A', 'B']],
            'skewness': [[processed_moments_2[label]['moments'][f'{ch}_skewness'] for label in labels_dict["Light Moments"]] for ch in ['L', 'A', 'B']]
        }
    }


    # Define markers for each channel and dictionary
    markers = {
        "Shadow Moments": ['o', '^', 's'],  # L, A, B for Shadow Moments
        "Light Moments": ['x', 'D', 'v']   # L, A, B for Light Moments
    }

    # Titles for the plots
    titles = ['Mean for L, A, B Channels', 'Variance for L, A, B Channels', 'Skewness for L, A, B Channels']

    _, axes = plt.subplots(3, 1, figsize=(18, 10))

    # Iterate over each moment type (mean, variance, skewness)
    for i, moment_type in enumerate(['mean', 'variance', 'skewness']):
        for dict_name in ['Shadow Moments', 'Light Moments']:
            for channel in range(3):  # 0: L, 1: A, 2: B
                for cluster_idx in range(len(moments_dict[dict_name][moment_type][channel])):
                    axes[i].scatter(cluster_idx, moments_dict[dict_name][moment_type][channel][cluster_idx], 
                                    color=colors_dict[dict_name][cluster_idx], 
                                    marker=markers[dict_name][channel],
                                    label=f"{dict_name}: {['L', 'A', 'B'][channel]} channel {moment_type} (Cluster {cluster_idx})")
        
        
        axes[i].set_title(titles[i])
        axes[i].set_xlabel('Label')
        axes[i].set_ylabel(f"{moment_type.capitalize()} Value")
        if i == 0:
            axes[i].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
        
    plt.tight_layout()
    plt.show()
    
def display_local_shadow_regions_enhancement(image, corrected_image, enhanced_image, image_name, figsize=(10, 5)):
    """
    Displays the initial shadow regions and the corresponding corrected image after shadow removal side by side.
    :param image: np.ndarray, The original image with shadow regions.
    :param corrected_image: np.ndarray, The corrected image where shadows have been removed.
    :param enhanced_image: np.ndarray, The enhanced image where shadows have been removed.
    :param image_name: str, A string representing the name or identifier of the image, which will be used in the titles.
    :param figsize: tuple(int), default (10,5), A tuple specifying the size of the figure.
    """
    _, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].imshow(image)
    axes[0].set_title(f'Image: {image_name}')
    axes[0].axis('off')

    axes[1].imshow(corrected_image)
    axes[1].set_title(f'Initial corrected image')
    axes[1].axis('off')
    
    axes[2].imshow(enhanced_image)
    axes[2].set_title(f'Enhanced corrected image')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()
    
def display_final_corrected_image(image, initial_corrected_image, enhanced_image, final_corrected_image, image_name, figsize=(10, 5)):
    """
    Displays the initial shadow regions and the corresponding corrected image after shadow removal side by side.
    :param image: np.ndarray, The original image with shadow regions.
    :param initial_corrected_image: np.ndarray, The corrected image where shadows have been removed.
    :param enhanced_image: np.ndarray, The enhanced image where shadows have been removed.
    :param final_corrected_image: np.ndarray, The corrected image where shadows have been removed.
    :param image_name: str, A string representing the name or identifier of the image, which will be used in the titles.
    :param figsize: tuple(int), default (10,5), A tuple specifying the size of the figure.
    """
    _, axes = plt.subplots(1, 4, figsize=figsize)

    axes[0].imshow(image)
    axes[0].set_title(f'Image: {image_name}')
    axes[0].axis('off')

    axes[1].imshow(initial_corrected_image)
    axes[1].set_title(f'Initial corrected image')
    axes[1].axis('off')
    
    axes[2].imshow(enhanced_image)
    axes[2].set_title(f'Enhanced corrected image')
    axes[2].axis('off')
    
    axes[3].imshow(final_corrected_image)
    axes[3].set_title(f'Final corrected image')
    axes[3].axis('off')

    plt.tight_layout()
    plt.show()
    
def display_shadow_mask_and_final_corrected_image(image, mask, final_corrected_image, image_name, figsize=(10, 5)):
    """
    Displays the initial shadow regions, shadow mask and the corresponding corrected image after shadow removal side by side.
    :param image: np.ndarray, The original image with shadow regions.
    :param initial_corrected_image: np.ndarray, The corrected image where shadows have been removed.
    :param mask: np.ndarray, The binary mask corresponding to the image, which will be displayed in grayscale.
    :param final_corrected_image: np.ndarray, The corrected image where shadows have been removed.
    :param image_name: str, A string representing the name or identifier of the image, which will be used in the titles.
    :param figsize: tuple(int), default (10,5), A tuple specifying the size of the figure.
    """
    _, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].imshow(image)
    axes[0].set_title(f'Image: {image_name}')
    axes[0].axis('off')
    
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title(f'Mask: {image_name}')
    axes[1].axis('off')
    
    axes[2].imshow(final_corrected_image)
    axes[2].set_title(f'Final corrected image')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

def plot_metrics(metrics_data):
    """
    Plot bar plots for each metric (SRI, CD, GMSD) for each cluster with random colors.

    :param metrics_data: Dictionary containing the metrics data for each cluster.
                         Format: {"SRI": {cluster_id: value, ...}, 
                                  "CD": {cluster_id: value, ...}, 
                                  "GMSD": {cluster_id: value, ...}}
    """
    # Define subplots
    metrics = ["SRI", "CD", "GMSD"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(25, 8), sharey=False)

    for i, metric in enumerate(metrics):
        ax = axes[i]
        data = metrics_data.get(metric, {})

        # Extract cluster labels and values
        cluster_labels = list(data.keys())
        values = list(data.values())

        # Generate a random color for the bar plot
        random_color = [random.random() for _ in range(3)]

        # Create the bar plot with the random color
        ax.bar(cluster_labels, values, color=random_color)
        ax.set_title(f"{metric} per Cluster")
        ax.set_xlabel("Cluster")
        ax.set_ylabel(metric)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        # Ensure the x-axis ticks and labels correspond to available clusters
        ax.set_xticks(range(len(cluster_labels)))
        ax.set_xticklabels(cluster_labels, rotation=90, ha="right")

    plt.tight_layout()
    plt.show()

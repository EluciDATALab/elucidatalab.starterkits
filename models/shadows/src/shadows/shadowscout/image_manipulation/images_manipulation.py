import numpy as np
import cv2
from scipy.ndimage import label
import torch

EPSILON = 1e-6

def RGB_to_HSI(RGB_image):
    """
    Converts an image in RGB to HSI.
    :param RGB_image: numpy array, the image in RGB
    :return: numpy array, image in HSI
    """
    # Convert RGB to float32
    RGB_image = RGB_image.astype(np.float32) / 255.0

    # Separate the RGB channels
    R, G, B = cv2.split(RGB_image)

    # Calculate Intensity
    I = np.divide(R + G + B, 3)

    # Calculate Saturation
    min_RGB = np.minimum(np.minimum(R, G), B)
    S = 1 - (3 / (R + G + B + 1e-6) * min_RGB)

    # Avoid dividing by zero
    S[I == 0] = 0

    # Calculate Hue
    numerator = 0.5 * ((R - G) + (R - B))
    denominator = np.sqrt((R - G) ** 2 + (R - B) * (G - B))
    theta = np.arccos(numerator / (denominator + EPSILON))

    H = theta
    H[B > G] = 2 * np.pi - H[B > G]
    H = H / (2 * np.pi)

    # Combine H, S, and I back into one image
    HSI = cv2.merge((H, S, I))
    return HSI


def generate_H_I_channel(H, I):
    """
    Generates a H-I channel from the H and I channels.
    :param H: numpy array, the Hue channel
    :param I: numpy array, the Intensity channel
    :return: numpy array, the H-I channel
    """
    h_i = H.astype(float) / (I.astype(float) + EPSILON)
    h_i[h_i > 1] = np.log(h_i[h_i > 1]) + 1

    return h_i

def apply_gaussian_blur(channel, sigma):
    """
    Applies gaussian blur to channel
    :param channel: numpy array
    :param sigma: float: the sigma value for the appropriate Gaussian kernel size
    :returns channel with gaussian blur
    """

    size = get_gaussian_kernel_size(sigma)
    kernel = (size, size)
    return cv2.GaussianBlur(channel, kernel, sigma)

def generate_channels(image, sigma=0.5):
    """
    Gemerates the H-I, I, S and H channels for the given image.
    :param image: numpy array, the image in RGB
    :param gaussian_kernel_sigma: float, default 0.5, the sigma value for the appropriate Gaussian kernel size
    :return: H_I, I, S, H channels as numpy arrays
    """    
    # RGB to HSI
    image_HSI = RGB_to_HSI(image)
    HSI = [apply_gaussian_blur(c, sigma) for c in cv2.split(image_HSI)]

    image_HSI = cv2.merge(HSI)
            
    # Calculate H-I
    H_I = generate_H_I_channel(H=image_HSI[:, :, 0], I=image_HSI[:, :, -1])   

    # return H_I, I, S, H
    return H_I, image_HSI[:, :, -1], image_HSI[:, :, 1], image_HSI[:, :, 0]


def get_gaussian_kernel_size(sigma):
    """
    Determines an appropriate Gaussian kernel size given a sigma.
    Typically, the kernel size should be 6 times sigma to cover 3 standard deviations to each side.
    :param sigma: int, the value of sigma
    :return: int, kernel size
    """
    size = 2 * int(3 * sigma) + 1
    return size


def generate_combined_mask(channels, thresholds, channel_weights, threshold_direction, convert_to_binary=True):
    """
    Generates the combined mask (output).
    :param channels: list. List of numpy arrays with H-I, I and S channels (or more)
    :param thresholds: list. List of thresholds per channels
    :param threshold_direction: list. List indicating, for each channel, whether values
    above the threshold should be shadows (direction='above') or values below (direction='below').
    Default=None (will be set to ['above', 'below', 'above'] for HI, I and S channels)
    :return: numpy array, the combined mask (the final output)
    """
    cv2_thresholds = {1: cv2.THRESH_BINARY, -1: cv2.THRESH_BINARY_INV}
    
    n_channel_weights = (channel_weights/channel_weights.sum())
    combined_mask = np.zeros_like(channels[0])
    masks = []
    for channel, threshold, channel_weight, thr_direction in zip(channels, thresholds, n_channel_weights, threshold_direction):
        _, thresholded_mask = cv2.threshold(channel, threshold, 1, cv2_thresholds[thr_direction])
        combined_mask += np.float32(thresholded_mask) * channel_weight
        masks.append(thresholded_mask)

    if convert_to_binary:
        combined_mask = combined_mask > 0.5

    return combined_mask, masks


def generate_combined_mask_torch(channels, thresholds, channel_weights, threshold_direction, convert_to_binary=True):
    """
    Generates the combined mask (output) using PyTorch tensors.
    :param channels: list. List of PyTorch tensors with H-I, I and S channels (or more).
    :param thresholds: list. List of thresholds per channel (scalar or tensor on the same device as `channels`).
    :param channel_weights: PyTorch tensor. Learnable weights for the channels (on the same device as `channels`).
    :param threshold_direction: list. List indicating, for each channel, whether values
    above the threshold should be shadows (direction='above') or values below (direction='below').
    :param convert_to_binary: bool. Whether to convert the combined mask to binary.
    :return: PyTorch tensor, the combined mask, and a list of masks for each channel.
    """
    cv2_thresholds = {1: cv2.THRESH_BINARY, -1: cv2.THRESH_BINARY_INV}
    
    # Normalize channel weights
    n_channel_weights = channel_weights / channel_weights.sum()

    # Combined mask initialization
    device = channels[0].device  # Ensure all operations are on the same device
    combined_mask = torch.zeros_like(channels[0], dtype=torch.float32, device=device)
    masks = []

    for channel, threshold, channel_weight, thr_direction in zip(channels, thresholds, n_channel_weights, threshold_direction):
        # Convert PyTorch tensor to NumPy for OpenCV, then back to PyTorch
        channel_np = channel.detach().cpu().numpy()
        _, thresholded_mask_np = cv2.threshold(channel_np, threshold.item(), 1, cv2_thresholds[thr_direction])
        thresholded_mask = torch.tensor(thresholded_mask_np, dtype=torch.float32, device=device)
        
        # Update combined mask
        combined_mask += thresholded_mask * channel_weight
        masks.append(thresholded_mask)

    # Convert to binary if requested
    if convert_to_binary:
        combined_mask = (combined_mask > 0.5).float()

    return combined_mask, masks

def eliminate_small_low_brightness_objects(combined_mask, n_pixels):
    """
    Eliminates small low brightness objects. Step 1 f post processing results.
    :param combined_mask: numpy array, the combined mask from the H-I, I and S channels
    :param n_pixels: int, the spatial lower limit in pixels
    :return: numpy array, the filtered mask
    """
    # Label connected components
    labeled_array, _ = label(combined_mask)

    # Count the size of each component
    sizes = np.bincount(labeled_array.ravel())

    # Create a mask to remove small components
    # +1 offset in sizes array because bincount indexes start at 0
    remove_small_areas_mask = sizes > n_pixels

    # The first component (background) is not to be removed
    remove_small_areas_mask[0] = False

    # Filter out small components
    filtered_mask = remove_small_areas_mask[labeled_array]
    
    return filtered_mask

def eliminate_small_bright_ground_objects_in_shadow(filtered_mask, kernel_size=3):
    """
    Eliminates small bright ground objects in shadows. Step 2 of post processing results.
    :param filtered_mask: numpy array, the filtered mask from the eliminate_small_low_brightness_objects function
    :param kernel_size: int, default value 3, the kernel size for the closing operation
    :return: numpy array, the closed mask
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    # Perform the closing operation
    closed_mask = cv2.morphologyEx(filtered_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    
    return closed_mask

def postprocess_shadow_detection(mask, kernel_size_ground_objects=3, n_pixels_brightness=130):
    """
    Wrapper to implement the two steps of the post-processing
    :param mask: numpy array, the filtered mask from the eliminate_small_low_brightness_objects function
    :param kernel_size: int, default value 3, the kernel size for the closing operation. Default=3
    :param n_pixels: int, the spatial lower limit in pixels. Default=130
    :return: numpy array, the post-processed shadow mask
    """

    return eliminate_small_bright_ground_objects_in_shadow(
        eliminate_small_low_brightness_objects(mask, n_pixels_brightness),
        kernel_size_ground_objects)
    

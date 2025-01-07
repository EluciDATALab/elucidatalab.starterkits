import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt as bwdist
from scipy.ndimage import gaussian_filter


def compute_metrics(combined_mask, image_val):
    """
    Computes the metrics of the results.
    The metrics are accuracy, precision, recall, F1 score and Jaccard distance.
    :param combined_mask: numpy array, the final output
    :param image_val: numpy array, the validation image
    return: (float, float, float, float, float), the accuracy, precision, recall, F1 score and Jaccard distance
    """
    TP = np.sum((combined_mask == 1) & (image_val == 255))
    FP = np.sum((combined_mask == 1) & (image_val == 0))
    TN = np.sum((combined_mask == 0) & (image_val == 0))
    FN = np.sum((combined_mask == 0) & (image_val == 255))

    accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    intersection = np.logical_and(image_val == 255, combined_mask == 1)
    union = np.logical_or(image_val == 255, combined_mask == 1)

    jaccard_similarity = np.sum(intersection) / np.sum(union)
    jaccard_distance = 1 - jaccard_similarity

    return accuracy, precision, recall, f1_score, jaccard_distance

def compute_BER(combined_mask, image_val):
    """
    Computes the Balanced Error Rate of the mask.
    :param combined_mask: numpy array, the final output
    :param image_val: numpy array, the validation image
    return: float, BER
    """
    TP = np.sum((combined_mask == 1) & (image_val == 1))
    FP = np.sum((combined_mask == 1) & (image_val == 0))
    TN = np.sum((combined_mask == 0) & (image_val == 0))
    FN = np.sum((combined_mask == 0) & (image_val == 1))
    
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    
    return (1 - 0.5 * (recall + specificity)) * 100

def compute_F1_omega_score(combined_mask, image_val):
    """
    Computes the F1 omega score of the mask.
    :param combined_mask: numpy array, the final output
    :param image_val: numpy array, the validation image
    return: float, F1 omega score
    """
    image_val = image_val.astype(bool)
    
    d_image_val = image_val.astype(np.float64)  # Use double for computations
    
    E = np.abs(combined_mask - d_image_val)
    
    Dst, IDXT = bwdist(d_image_val, return_indices=True)
    
    # Pixel dependency
    K = cv2.getGaussianKernel(7, 5)
    K = K * K.T  # Create 2D Gaussian kernel
    Et = E.copy()
    Et[~image_val] = Et[IDXT[0][~image_val], IDXT[1][~image_val]]  # To deal correctly with the edges of the foreground region
    EA = gaussian_filter(Et, sigma=5)
    MIN_E_EA = E.copy()
    MIN_E_EA[image_val & (EA < E)] = EA[image_val & (EA < E)]
    
    # Pixel importance
    B = np.ones_like(image_val, dtype=np.float64)
    B[~image_val] = 2 - 1 * np.exp(np.log(1 - 0.5) / 5 * Dst[~image_val])
    Ew = MIN_E_EA * B
    
    TPw = np.sum(d_image_val) - np.sum(Ew[image_val])
    FPw = np.sum(Ew[~image_val])
    
    R = 1 - np.mean(Ew[image_val])  # Weighed Recall
    P = TPw / (np.finfo(float).eps + TPw + FPw)  # Weighted Precision
    
    F = (2) * (R * P) / (np.finfo(float).eps + R + P)  # Beta=1
    # F = (1 + Beta^2) * (R * P) / (eps + R + (Beta * P))
    
    return F

def compute_F1_score(combined_mask, image_val):
    """
    Computes the F1 score of the mask.
    :param combined_mask: numpy array, the final output
    :param image_val: numpy array, the validation image
    return: float, F1 score
    """
    TP = np.sum((combined_mask == 1) & (image_val == 1))
    FP = np.sum((combined_mask == 1) & (image_val == 0))
    FN = np.sum((combined_mask == 0) & (image_val == 1))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return f1_score


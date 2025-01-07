import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
from skimage.measure import find_contours


def plot_shadow_polygons(image, shadow_mask, ax=None, verbose=False):
    """Plot image with shadow contours overlayed

    :param image: array. Image
    :param shadow_mask: array. Shadow mask array
    :param simplify_tolerance
    """

    # find shadow contours
    contours = find_contours(shadow_mask == 1)
    
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = plt.gca()
    else:
        fig = plt.gcf()
    ax.imshow(image)
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], color="red", linewidth=2)
            
    return fig, contours

import torch
from torchvision.transforms import functional as F

class JointTransform:
    """
    Applies a series of transformations to both an image and its corresponding mask.
    """
    def __init__(self, transforms):
        """
        :param transforms: list of transformations.
        """
        self.transforms = transforms

    def __call__(self, img):
        """
        Iterates through each transformation and applies them to both the image and mask, returning the transformed image and mask.
        :param img: numpy array, the image
        :returns numpy.Array, the transformed image
        """
        for t in self.transforms:
            img = t(img)
        return img

class RandomHorizontalFlip:
    """
    Randomly flip an image and its mask horizontally.
    """
    def __call__(self, img):
        """
        With a 50% chance, it horizontally flips the image and mask.
        :param img: numpy array, the image
        :returns numpy.Array, the transformed image.
        """
        if torch.rand(1) < 0.5:
            img = F.hflip(img)
        return img

class RandomVerticalFlip:
    """
    Randomly flip an image and its mask vertically.
    """
    def __call__(self, img):
        """
        With a 50% chance, it vertically flips the image and mask.
        :param img: numpy array, the image
        :returns numpy.Array, the transformed image.
        """
        if torch.rand(1) < 0.5:
            img = F.vflip(img)
        return img
    
class JointTransformForCorrection:
    """
    Applies a series of transformations to both an image and its corresponding mask.
    """
    def __init__(self, transforms):
        """
        :param transforms: list of transformations.
        """
        self.transforms = transforms

    def __call__(self, image, inpainted_image, mask):
        """
        Iterates through each transformation and applies them to both the image and mask, returning the transformed image and mask.
        :param img: numpy array, the image
        :returns numpy.Array, the transformed image
        """
        for t in self.transforms:
            image, inpainted_image, mask = t(image, inpainted_image, mask)
        return image, inpainted_image, mask
    
class RandomHorizontalFlipForCorrection:
    """
    Randomly flip an image and its mask horizontally.
    """
    def __call__(self, img, impainted_img, mask):
        """
        With a 50% chance, it horizontally flips the image and mask.
        :param img: numpy array, the image
        :returns numpy.Array, the transformed image.
        """
        if torch.rand(1) < 0.5:
            img = F.hflip(img)
            impainted_img = F.hflip(impainted_img)
            mask = F.hflip(mask)
        return img, impainted_img, mask

class RandomVerticalFlipForCorrection:
    """
    Randomly flip an image and its mask vertically.
    """
    def __call__(self, img, impainted_img, mask):
        """
        With a 50% chance, it vertically flips the image and mask.
        :param img: numpy array, the image
        :returns numpy.Array, the transformed image.
        """
        if torch.rand(1) < 0.5:
            img = F.vflip(img)
            impainted_img = F.vflip(impainted_img)
            mask = F.vflip(mask)
        return img, impainted_img, mask
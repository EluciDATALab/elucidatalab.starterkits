import numpy as np
from scipy.stats import pearsonr
from pathlib import Path

import cv2
import rasterio
from PIL import Image

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from shadows.shadowscout.data.data_augmentation import JointTransform, RandomHorizontalFlip, RandomVerticalFlip
from shadows.shadowscout.data.utils import calculate_min_max_for_channels
from shadows.shadowscout.image_manipulation.images_manipulation import generate_channels, apply_gaussian_blur

import warnings
from rasterio.errors import NotGeoreferencedWarning
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


class ShadowDataset(Dataset):
    """
    This class is a custom PyTorch dataset designed for handling shadow images with optional masks and transformations. 
    It allows for loading, preprocessing, and augmenting images and masks.
    """
    def __init__(self, image_paths, image_shape, kernel_sigma, transform=None, number_of_channels=3):
        """
        Initializes the dataset with image paths, resize value, 
        kernel sigma, optional mask paths, a transform function, and a flag for a special dataset format.
        :param image_paths: list(str), List of paths to the images.
        :param image_shape: list, Shape of images
        :param kernel_sigma: float, Sigma value for generating image channels.
        :param transform: JointTransform, default None, Optional transformation function to be applied to the images and masks.
        :param number_of_channels: int. Number of channels to retain. If None, use all available (first 3 are converted to HI, I, S)
        """
        self.image_paths = image_paths
        self.image_shape = image_shape
        self.kernel_sigma = kernel_sigma
        self.transform = transform
        self.to_tensor = transforms.ToTensor()
        self.number_of_channels = number_of_channels
        
    def __len__(self):
        """
        Returns the number of images in the dataset.
        :return: int, dataset length
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Fetches and processes an image (and its mask if available) based on the index idx.
        Loads the image. Resizes the image to the specified image_shape.
        If transform is provided, applies the transformations to the image and the mask/NIR band.
        Generates the image channels (H_I, I, S).
        Returns the processed and transformed image tensors.
        :param idx: int, image index
        :return: (Tensor, Tensor) the processed and transformed image tensors
        """
        image_path = self.image_paths[idx]
        
        # Shadow image
        image = read_and_resize(image_path, self.image_shape)
        
        # Data Augumentation
        if self.transform:
            # Convert to PIL for augmentation
            image = Image.fromarray(image.astype(np.uint8))
            image = np.array(self.transform(image))
            
        # generate channels
        channels = [self.to_tensor(i.astype(np.float32))
                     for i in generate_channels(image=image[:, :, :3], sigma=self.kernel_sigma)[:3]]
        if (image.shape[2] > 3) & (self.number_of_channels > 3):
            channels += [self.to_tensor(apply_gaussian_blur(image[:, :, i], self.kernel_sigma).astype(np.float32))
                         for i in np.arange(3, self.number_of_channels)]
        
        return channels
    

def read_and_resize(image_path, image_shape):        
    """Read and resize images
    
    :param image_path: str. Path to image
    :param image_shape: tuple. first and second dimensions of image
    :return: image
    """
    im_channels = []
    with rasterio.open(image_path) as src:
        im_channels = [src.read(i + 1) for i in range(src.count)]
    image = np.dstack(im_channels)
    
    # Resize
    image = cv2.resize(image, image_shape, interpolation=cv2.INTER_LINEAR)

    return image


def prepare_datasets(dataset_dir, config, transform_image=True):
    """Prepare datasets for model training
    
    :param dataset_dir: dict. Dictionary with directories of train/val/test images
    :param config: dict. Dictionary with model configuration parameters
    :param transform_image: bool. Whether to apply transformations on the training dataset. Default=True
    :return image_paths: dict. Dictionary with train/val/test image paths
    """
    #### Collect paths of images of train, validate and test
    image_paths = {k: None for k in dataset_dir.keys()}
    for key, value in dataset_dir.items():
        if (config['full_dataset']) & (key == 'train'):
            directories_ = [Path(d) for d in list(dataset_dir.values())]
        else:
            directories_ = [Path(value)]
        image_paths[key] = np.concatenate([np.array(list(d.glob('*'))).astype(str)
                                           for d in directories_])

    #### Define the data augmentation pipeline
    transform = JointTransform([
        RandomHorizontalFlip(),
        RandomVerticalFlip()])

    #### Create the datasets
    data = {k: {} for k in image_paths.keys()}
    for key, value in image_paths.items():
        data[key]['dataset'] = ShadowDataset(
            value, 
            config['img_shape'], 
            config['gaussian_kernel_sigma'],
            transform=transform if (key == 'train') & (transform_image) else None,
            number_of_channels=config.get('number_of_channels', 3))
        if key == 'train':
            # collect stats from train dataset
            data[key]['stats'] = (
                calculate_min_max_for_channels(data[key]['dataset']))
            data[key]['dataloader'] = DataLoader(
                data[key]['dataset'], 
                batch_size=config['batch_size'],
                shuffle=False)
        else:
            # define DataLoader object
            data[key]['dataloader'] = DataLoader(
                data[key]['dataset'], 
                batch_size=config['val_batch_size'],
                shuffle=False)

            
    return image_paths, data


def determine_threshold_direction(dataloader, n_images, number_of_channels):
    """Determine relationship between channel and shadows

    :param dataloader: torch dataloader. Model dataloader
    :param n_images: int. Number of images to sample.
    :param number_of_channels: int. Number of channels to use

    :return list with length=number_of_channels indicating whether a channel
    is positively (1) or negatively (-1) correlated with shadows
    """

    max_images = len(dataloader.dataset)
    rnd_images = np.random.choice(n_images, 
                                  max_images if 20 > max_images else n_images, 
                                  replace=False)
    thr_directions = []
    for r in rnd_images:
        hi = dataloader.dataset[r][0].flatten()
        thr_directions_ = np.empty(number_of_channels)
        for k, c in enumerate(dataloader.dataset[r]):
            thr_directions_[k] = pearsonr(hi, c.flatten())[0]
        thr_directions.append(thr_directions_)
    return [int(t) for t in np.sign(np.median(np.stack(thr_directions), 0))]
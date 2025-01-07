import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

from shadows.shadowscout.image_manipulation.images_manipulation import get_gaussian_kernel_size

class ShadowScout(nn.Module):
    """
    Class for unsupervised CNN for shadow detection
    """
    def __init__(self,
                 image_shape, 
                 kernel_sigma=0.5,
                 dropout_rate=0.3,
                 channel_weights=[0.6, 0.2, 0.2],
                 number_of_channels=3,
                 weights_clamp_range=[.1, .9],
                 learnable_weights=True):
        """
        Initializes the network's layers and parameters.
        :param kernel_sigma: float, Used to determine the kernel size for convolutional layers.
        :param image_shape: tuple, Shape of the input images.
        :param dropout_rate: float, default 0.3, Dropout rate for regularization.
        :param number_of_channels: int, default 3, Number of channels of the input images (3 for RGB, 4 for RGB + NIR).
        :param weights_clamp_range: list. List of lower and upper bounds for weights. Default=[.1, .9]
        :param learnable_weights: bool. Whether weights are learnable parameters. Default=True
        """
        super(ShadowScout, self).__init__()
        self.conv1 = nn.Conv2d(number_of_channels, 16, kernel_size=get_gaussian_kernel_size(kernel_sigma), padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout2d(dropout_rate)
        self.pool1 = nn.MaxPool2d(2)  # Reduces size to 256x256
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=get_gaussian_kernel_size(kernel_sigma), padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout2d(dropout_rate)
        self.pool2 = nn.MaxPool2d(2)  # Reduces size to 128x128
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=get_gaussian_kernel_size(kernel_sigma), padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout2d(dropout_rate)
        self.pool3 = nn.MaxPool2d(2)  # Reduces size to 64x64
        
        self.fc1 = nn.Linear(64 * (image_shape[0] // 8) * (image_shape[0] // 8), 256)  # Updated number of input features
        self.bn4 = nn.BatchNorm1d(256)
        self.act4 = nn.ReLU()
        self.drop4 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(256, number_of_channels) # Three output values for H_I, I, S thresholds
        self.fc2_sigmoid = nn.Sigmoid()
        
        self.learnable_weights = learnable_weights
        if learnable_weights:
            self.channel_weights = AlphaWeight(channel_weights, clamp_range=weights_clamp_range)  # Example initial guesses
        else:
            self.channel_weights = np.array(channel_weights)
        
        self.apply(self._init_weights)  # Apply weight initialization
        self.config = {}
        
    def _init_weights(self, m):
        """
        Initializes the weights of the network layers.
        :param m: Layer
        """
        if isinstance(m, nn.Conv2d):
            # Kaiming (He) initialization for convolutional layers
            init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            # Kaiming (He) initialization for fully connected layers
            init.kaiming_normal_(m.weight, nonlinearity='relu')
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            # BatchNorm layers: weights initialized to 1, biases to 0
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)

    def forward(self, inputs):
        """
        Defines the forward pass of the network.
        :param x: Layer
        """
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.drop1(x)
        x1 = self.pool1(x)

        x = self.conv2(x1)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.drop2(x)
        x2 = self.pool2(x)

        x = self.conv3(x2)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.drop3(x)
        x3 = self.pool3(x)

        x = torch.flatten(x3, 1)
        x = self.fc1(x)
        x = self.bn4(x)
        x = self.act4(x)
        x4 = self.drop4(x)
        
        delta_thresholds = self.fc2(x4) 
        thresholds = torch.sigmoid(delta_thresholds) 
        return thresholds

    def save_state_dict(self, path):
        """Save model state_dict. config file will be added to it
        
        :param path: str. Path to save state_dict to
        """
        state_dict = self.state_dict()
        state_dict['config'] = self.config  # Add custom dict to state_dict
        torch.save(state_dict, path)

    def load_state_dict(self, path, map_location):
        """load model state_dict
        
        :param path: str. Path to save state_dict to
        :param map_location: str. device to load state_dict on to
        """
        state_dict = torch.load(path, map_location=map_location)
        # Extract the custom dict if it exists
        self.config = state_dict.pop('config', {})
        super(ShadowScout, self).load_state_dict(state_dict)


import torch
import torch.nn as nn

class AlphaWeight(nn.Module):
    """Define learnable weights parameter with individual range constraints."""
    def __init__(self, initial_values=[.5, .5, .5], clamp_range=[.1, .9]):
        """
        Initializes learnable parameter
        
        :param initial_values: [array, list]. array or list of length 3 with initial weight values.
        Default=[.5, .5, .5]
        :param clamp_range: [list of tuples]. List of tuples specifying the range of values for each weight.
        Default=[(.1, .9), (.1, .9), (.1, .9)]
        """
        super(AlphaWeight, self).__init__()
        # if len(initial_values) != len(clamp_range):
        #     raise ValueError("Length of initial_values must match the length of clamp_range.")
        
        self.parameter = nn.Parameter(
            torch.stack([torch.Tensor([i]) for i in initial_values]))  # Initial values

        if (isinstance(clamp_range[0], (int, float))) & (len(clamp_range) == 2):
            clamp_range = [clamp_range] * len(initial_values)
        
        self.clamp_ranges = clamp_range

    def constrain(self):
        """Apply constraints to each parameter value based on its range."""
        for i, (low, high) in enumerate(self.clamp_ranges):
            self.parameter.data[i].clamp_(low, high)

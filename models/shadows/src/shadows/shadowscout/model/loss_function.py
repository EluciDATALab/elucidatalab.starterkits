import torch
import torch.nn as nn
from shadows.shadowscout.image_manipulation.images_manipulation import generate_combined_mask_torch

class CombinedMaskInterClassVarianceLoss(nn.Module):
    """
    This class defines a custom loss function for a neural network that combines interclass variance with regularization terms. 
    It is designed to optimize threshold values for three image channels (H_I, I, S),
    while incorporating L1 and L2 regularization on the model parameters.
    """
    def __init__(self, alpha_H_I=0.6, alpha_I=0.2, alpha_S=0.2, l1_lambda=1e-5, l2_lambda=1e-4):
        """
        Initializes the loss function with specific weights for each channel and regularization parameters.
        :param alpha_HI: float, default 0.6, Weight for the variance contributions of the H_I channel.
        :param alpha_I: float, default 0.2, Weight for the variance contributions of the I channel.
        :param alpha_S: float, default 0.2, Weight for the variance contributions of the S channel.
        :param l1_lambda: float, default 1e-5, Regularization weight for L1 norm.
        :param l2_lambda: float, default 1e-4, Regularization weight for L2 norm.
        """
        super().__init__()
        self.alpha_H_I = alpha_H_I
        self.alpha_I = alpha_I
        self.alpha_S = alpha_S
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

    def weighted_mean_variance(self, variances, weights):
        """Calculate weighted mean of the ICV per channel
        :param variances: tensor. Tensor with ICV per channel
        :param weights: tensor. Weights per channel
        :return tensor: weighted variance average
        """
        weighted_sum = torch.stack(
                [value * weight
                 for value, weight in zip(variances, weights)]).sum()
        total_weight = weights.sum()
        weighted_variance = weighted_sum / total_weight

        return weighted_variance

    def forward(self, thresholds, inputs, model, threshold_direction):
        """
        Computes the loss given the thresholds, image channels, and model.
        Broadcasts thresholds to match the dimensions of the image channels.
        Generates masks for each channel using a hyperbolic tangent function.
        Combines the masks using element-wise minimum operations.
        Computes the interclass variance for each channel.
        Computes a weighted sum of variances for the channels.
        Adds L1 and L2 regularization terms based on the model parameters.
        :param thresholds: Tensor(list), Threshold values for each channel.
        :param H_I: Tensor, H-I channel
        :param I: Tensor, I channel
        :param S: Tensor, S channel
        :param model: ShadowNet, model
        :param threshold_direction: list. For each channel 1 (higher values correspond to shadows) or 0 (lower values
        correspond to shadows). Default: None (use [1, -1, 1] for HI, I and S)
        :returns: Tensor(float), the negative mean of the weighted variances, plus regularization terms, for optimization.
        """

        # prepare inputs and masks
        inputs_ = [[]] * model.number_of_channels
        input_masks = [[]] * model.number_of_channels
        for i in range(len(thresholds)):
            # formats masks and thresholds. I
            inputs__, input_masks_ = get_channels_and_masks(
                [inputs[k][i] for k in range(model.number_of_channels)], 
                thresholds[i],
                None,
                threshold_direction)
            for channel in range(model.number_of_channels):
                inputs_[channel] = inputs_[channel] + [inputs__[channel]]
                input_masks[channel] = input_masks[channel] + [input_masks_[channel]]

        # stack to make
        inputs_ = [torch.stack(w, dim=0) for w in inputs_]
        input_masks = [torch.stack(im, dim=0) for im in input_masks]
        
        # Combined mask made of H-I, I and S channel masks
        combined_mask = torch.min(torch.stack(input_masks[0:model.number_of_channels], dim=1), dim=1)[0]
        
        variances = [self.interclass_variance(i, combined_mask)
                     for i in inputs_]

        if model.learnable_weights:
            weighted_variance = self.weighted_mean_variance(variances, model.channel_weights.parameter)
        else:
            weighted_variance = self.weighted_mean_variance(variances, model.channel_weights)
        
        # Compute L1 and L2 regularization
        if self.l1_lambda != 0.0:
            l1_norm = sum(p.abs().sum() for p in model.parameters())
        else:
            l1_norm = 0.0
            
        if self.l2_lambda != 0.0:
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
        else:
            l2_norm = 0.0

        # Return negative mean of weighted variances for optimization
        return -weighted_variance + self.l1_lambda * l1_norm + self.l2_lambda * l2_norm

    def interclass_variance(self, images, masks):   
        """
        Calculates the interclass variance for a given image and mask.
        Calculates foreground and background weights.
        Computes mean values for the foreground and background regions.
        Calculates the proportions (P1 and P2) of foreground and background.
        Computes the interclass variance as the product of proportions and the squared difference between foreground and background means.
        :param images: Tensor, Input image channels.
        :param masks: tensor, Masks generated based on thresholds.
        :returns: Tensor(float), the inter-class variance
        """     
        foreground_weight = masks  # This is now a continuous value between 0 and 1
        background_weight = 1 - masks
        foreground_mean = (images * foreground_weight).sum(dim=(2, 3)) / foreground_weight.sum(dim=(2, 3)) + 1e-8
        background_mean = (images * background_weight).sum(dim=(2, 3)) / background_weight.sum(dim=(2, 3)) + 1e-8
        
        foreground_sizes = foreground_weight.sum(dim=(2, 3)) + 1e-8
        background_sizes = background_weight.sum(dim=(2, 3)) + 1e-8

        # Calculate proportions
        P1 = foreground_sizes / (foreground_sizes + background_sizes)
        P2 = 1 - P1

        # Calculate interclass variance
        variance = P1 * P2 * (foreground_mean - background_mean) ** 2
        return variance.median()  # Summing over pixels
    
class CombinedMaskCalinskiHarabaszLoss(nn.Module):
    """
    This class defines a custom loss function for a neural network that combines the Calinski-Harabasz Index with regularization terms.
    It is designed to optimize threshold values for three image channels (H_I, I, S),
    while incorporating L1 and L2 regularization on the model parameters.
    """
    def __init__(self, l1_lambda=1e-5, l2_lambda=1e-4):
        """
        Initializes the loss function with specific weights for each channel and regularization parameters.
        :param l1_lambda: float, default 1e-5, Regularization weight for L1 norm.
        :param l2_lambda: float, default 1e-4, Regularization weight for L2 norm.
        """
        super().__init__()
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

    def forward(self, thresholds, inputs, model, threshold_direction):
        """
        Computes the loss given the thresholds, image channels, and model.
        Broadcasts thresholds to match the dimensions of the image channels.
        Generates masks for each channel using a hyperbolic tangent function.
        Combines the masks using element-wise minimum operations.
        Computes the Calinski-Harabasz Index for each channel.
        Computes a weighted sum of indices for the channels.
        Adds L1 and L2 regularization terms based on the model parameters.
        :param thresholds: Tensor(list), Threshold values for each channel.
        :param H_I: Tensor, H-I channel
        :param I: Tensor, I channel
        :param S: Tensor, S channel
        :param model: ShadowNet, model
        :param threshold_direction: list. For each channel 1 (higher values correspond to shadows) or 0 (lower values
        correspond to shadows). Default: None (use [1, -1, 1] for HI, I and S)
        :returns: Tensor(float), the negative mean of the weighted indices, plus regularization terms, for optimization.
        """
        # Broadcast thresholds to match image dimensions
        ch_indices = []
        for i in range(thresholds.size(0)):
            channels_tmp = [input_[i] for input_ in inputs]
            ch_indices.append(
                self.get_calinski_harabasz_for_image(
                    thresholds[i], 
                    channels_tmp, 
                    model.channel_weights.parameter if model.learnable_weights else model.channel_weights,
                    threshold_direction))
            
        # Compute L1 and L2 regularization
        if self.l1_lambda != 0.0:
            l1_norm = sum(p.abs().sum() for p in model.parameters())
        else:
            l1_norm = 0.0
            
        if self.l2_lambda != 0.0:
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
        else:
            l2_norm = 0.0

        valid_ch_indices = [ch_idx for ch_idx in ch_indices if not torch.isnan(ch_idx)]
        if len(valid_ch_indices) / len(ch_indices) < .5:
            print('fewer than half of the samples in the batch have non-nan values, allowing nan to pass...')
            valid_ch_indices = ch_indices
        loss_value = -torch.median(torch.stack(valid_ch_indices)) + self.l1_lambda * l1_norm + self.l2_lambda * l2_norm
        
        # Return negative mean of weighted indices for optimization
        return loss_value
    
    def get_calinski_harabasz_for_image(self, thresholds, inputs, input_weights, threshold_direction):
        """ Calculate CH loss
        
        :param thresholds: list. List of thresholds for channels
        :param inputs: list. List of channels
        :param input_weights: list. List of channel weights
        :param threshold_direction: list. List of channel direction (relation to shadows)

        :return calinski-harabsz loss
        """
        
        weighted_inputs, input_masks = get_channels_and_masks(inputs, thresholds, input_weights, threshold_direction)
        
        flattened_weighted_inputs = torch.stack([i.flatten() for i in weighted_inputs], dim=1)
        
        if len(threshold_direction) == 3:
            combined_mask = input_masks[0]
            for mask, thr_direction in zip(input_masks[1:], threshold_direction[1:]):
                if thr_direction == 1:
                    combined_mask = torch.max(combined_mask, mask)[0]
            for mask, thr_direction in zip(input_masks[1:], threshold_direction[1:]):
                if thr_direction == -1:
                    combined_mask = torch.min(combined_mask, mask)[0] 
        else:
            combined_mask, _ = generate_combined_mask_torch(
                inputs, thresholds, input_weights, threshold_direction, convert_to_binary=False)
        
        flattened_combined_mask = torch.flatten(combined_mask)
        
        ch_index = self.calinski_harabasz_index(flattened_weighted_inputs, flattened_combined_mask)
        
        return torch.log(ch_index+1e-6)
        
    def calinski_harabasz_index(self, flattened_channels, flattened_mask):
        """
        Calculates the Calinski-Harabasz Index for a given image and mask.
        :param images: Tensor, Input flattened image channels.
        :param masks: Tensor, Masks generated based on thresholds.
        :returns: Tensor(float), the Calinski-Harabasz Index
        """
        
        foreground_weight = flattened_mask  # This is now a continuous value between 0 and 1
        background_weight = 1 - flattened_mask
        
        # Calculate the weighted mean for foreground and background
        foreground_mean = (flattened_channels * foreground_weight.view(-1, 1)).sum(dim=0) / (foreground_weight.sum() + 1e-8)
        background_mean = (flattened_channels * background_weight.view(-1, 1)).sum(dim=0) / (background_weight.sum() + 1e-8)

        overall_mean = flattened_channels.mean(dim=0)

        # Between-group sum of squares
        ssb = ((foreground_weight.sum() * (foreground_mean - overall_mean)**2).sum() + 
               (background_weight.sum() * (background_mean - overall_mean)**2).sum())

        # Within-group sum of squares
        ssw = ((foreground_weight.view(-1, 1) * (flattened_channels - foreground_mean)**2).sum(dim=1).view(-1, 1).sum() +
            (background_weight.view(-1, 1) * (flattened_channels - background_mean)**2).sum(dim=1).view(-1 , 1).sum())

        n_clusters = 2
        n_samples = flattened_channels.size(0)
        ch_index = (ssb / (n_clusters - 1)) / (ssw / (n_samples - n_clusters))

        return ch_index 


def get_channels_and_masks(inputs, thresholds, input_weights, threshold_direction):
    """Return weighted channels and channel masks and thresholds
    
    :param inputs: list. List of channels
    :param thresholds: list. List of channel thresholds
    :param input_weights: list. List of channel weights
    :param threshold_direction: list. List of channel direction (relation to shadows)

    :returns weighted_inputs: list. List of weighted channels
    :returns input_masks: list of channel makes
    """

    if len(threshold_direction) != len(thresholds):
        raise ValueError('threshold_direction and thresholds must have same lenghts')
    input_masks = []
    weighted_inputs = []
    for ix in range(len(thresholds)):
        input_ = inputs[ix]
            
        tmp_threshold = thresholds[ix]
        
        # make continuous mask adjusted for direction
        input_diff = (input_ - tmp_threshold) * threshold_direction[ix]
        
        # Vectorized mask generation 
        input_masks.append(torch.sigmoid(input_diff))

        # weight channel
        if input_weights is not None:
            input_ = input_ * input_weights[ix] / input_weights.sum()
        weighted_inputs.append(input_ )

    return weighted_inputs, input_masks

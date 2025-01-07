
import time
import csv
from fastprogress import progress_bar
import argparse
import json
import re
from pathlib import Path
import shutil
import cv2

import numpy as np
import pandas as pd
import random

import torch
from torch.nn.utils import clip_grad_norm_

from shadows.shadowscout.data.utils import normalize_channels
from shadows.shadowscout.data.datasets import prepare_datasets, determine_threshold_direction
from shadows.shadowscout.model.model import ShadowScout
from shadows.shadowscout.model.loss_function import CombinedMaskCalinskiHarabaszLoss, CombinedMaskInterClassVarianceLoss
from shadows.shadowscout.image_manipulation.images_manipulation import generate_combined_mask, postprocess_shadow_detection


def parse_inputs(inputs, stats, device, number_of_channels=3):
    """Normalize channels
    
    :param inputs: tuple with H_I, I, S channels and possibly NIR band
    :param number_of_channels: int. Number of channels to use. Default=3
    :returns normalized H_I, I, S channels and possibly NIR band
    """
    inputs_ = [inputs[i].to(device) for i in range(number_of_channels)]
    normalized_inputs = normalize_channels(
        inputs_, 
        stats)

    return normalized_inputs


def make_predictions(model, inputs, device):
    """Predict thresholds and calculate ICV loss

    :param model: torch model. 
    :param inputs: list with input channels
    :param device: str. 'cpu' or 'gpu'
    :return normalized channels
    :return thresholds
    :return loss
    """
    custom_loss = CombinedMaskCalinskiHarabaszLoss(
        l1_lambda=model.config['loss_l1_lambda'], l2_lambda=model.config['loss_l2_lambda'])
    normalized_inputs = parse_inputs(inputs, model.config['stats'], device,
                                     number_of_channels=model.config.get('number_of_channels', 3))
    if normalized_inputs[0].size(0) == 1:
        model_input = torch.cat([torch.cat(normalized_inputs, dim=1)] * 2)
        added_1 = True
    else:
        model_input = torch.cat(normalized_inputs, dim=1)
        added_1 = False

    thresholds = model(model_input)  # Model predicts thresholds
    if added_1:
        thresholds = thresholds[:1,:]
    loss = custom_loss(thresholds, normalized_inputs, model, model.config.get('threshold_direction', None)) # Calculate loss
    
    return normalized_inputs, thresholds, loss


def get_sklean_calinsky_harabasz_score(channels, mask, loss, weights):
    """Calculate Calinski-Harabasz.
    :param channels: list of normalized channels
    :param mask: array. 2-D mask array
    :param loss: CalinskiHarabasz loss object
    :param weights: list of channel weights
    :return float: calinski-harabasz score
    """
    
    weighted_channels = torch.stack([c * w / weights.sum() for c, w in zip(channels, weights)])
    
    return loss.calinski_harabasz_index(weighted_channels.permute(1, 2, 0).squeeze(), 
                                        mask.view(1, -1))
    
def get_interclass_variance_score(channels, mask, loss, weights):
    """Calculate InterClass variance.
    :param channels: list of normalized channels
    :param mask: array. 2-D mask array
    :param loss: CalinskiHarabasz loss object
    :param weights: list of channel weights
    :return float: interclass variance score
    """
    mask = mask.unsqueeze(0).unsqueeze(0)
    variances = [loss.interclass_variance(i.unsqueeze(0), mask) for i in channels]
    return float(loss.weighted_mean_variance(variances, weights))
    


def collect_metrics(normalized_inputs, thresholds, channel_weights, metrics_, config):
    """Collect metrics from epoch results
    
    :param normalized_inputs: list. List of normalized channels
    :param thresholds: array. Thresholds per channel
    :param channel_weights: list. List of channel weights
    :param metrics_: dict. Dictionary collecting metrics
    :param config: dict. Configuration parameters
    
    :return metrics_: dict. Dictionary collecting metrics
    :return detached_thresholds: array. Thresholds per channel
    :return output_masks: list. Predicted shadow masks
    """
    calinski_harabasz_loss = CombinedMaskCalinskiHarabaszLoss()
    interclassvariance_loss = CombinedMaskInterClassVarianceLoss()
    detached_thresholds = thresholds.cpu().detach() if torch.cuda.is_available() else thresholds.detach()
    detached_channel = [i.cpu().detach()
                        if torch.cuda.is_available() else i.detach()
                        for i in normalized_inputs]
    number_of_channels = len(normalized_inputs)
    output_masks = []
    for i in range(thresholds.size(0)):
        # get predicted mask
        combined_mask, _ = generate_combined_mask(
            [detached_channel[c][i].numpy().squeeze() for c in range(number_of_channels)],
            [float(detached_thresholds[i, c]) for c in range(number_of_channels)],
            channel_weights,
            threshold_direction=config.get('threshold_direction', None))
        
        # postprocess shadow mask
        postprocessed_mask = postprocess_shadow_detection(combined_mask)
        output_masks.append(postprocessed_mask)

        # check if all 0s
        if postprocessed_mask.sum() == 0:
            postprocessed_mask[0, 0] = 1e-8
        postprocessed_mask = torch.Tensor(postprocessed_mask)

        # get ICV
        icv = get_interclass_variance_score([d[i, :, :] for k, d in enumerate(detached_channel) if k < 3], 
                                            postprocessed_mask, 
                                            interclassvariance_loss, 
                                            channel_weights[:3])
        metrics_['variances'].append(icv)
        
        # get CH
        ch = get_sklean_calinsky_harabasz_score([d[i, :, :].view(1, -1) for k, d in enumerate(detached_channel) if k < 3], 
                                               postprocessed_mask, 
                                               calinski_harabasz_loss, 
                                               channel_weights[:3])
        metrics_['ch_score'].append(ch)
        
    return metrics_, detached_thresholds, output_masks


def log_epoch(log_metric_file_path, epoch, avg_metrics, model_time, channel_weights):
    """Log epoch results
    
    :param log_metric_file_path: str. Path to log file
    :param epoch: int. Epoch number
    :param avg_metrics: dict. Dictionary with average value per metric and per split
    :param model_time: dict. Dictionary with elapsed epoch time per split
    :param channel_weights: list. List of HI, I and S channel weights
    """
    # Check if the file exists
    if not log_metric_file_path.exists():
        columns = ['Epoch', 'Loss_Train', 'Loss_Validation',
                  'Interclass_variance_Train', 
                  'Interclass_variance_Validation',
                  'Calinsky-Harabasz_Train',
                  'Calinsky-Harabasz_Validate', 
                  'HI_weight', 'I_weight', 'S_weight']
        columns = columns + [f'Band{i+1}' for i in np.arange(3, len(channel_weights))]
        with open(log_metric_file_path, 'x') as file:
            writer_csv = csv.writer(file)
            writer_csv.writerow(columns)
    
    # add metrics to the csv log
    with open(log_metric_file_path, mode='a', newline='') as file:
        writer_csv = csv.writer(file)
        values = ([epoch] 
                  + [avg_metrics[split][m] for m in ['loss', 'variances', 'ch_score']
                     for split in ['train', 'validate']]
                  + channel_weights)
        writer_csv.writerow(values)
    log = ('Epoch: {:03d}, Train Loss: {:.4f}, '
           'Train Total Inter-class Variance: {:.4f}, Train Calinsky-Harabasz score: {:.4f}, '
           'Valid Loss: {:.4f}, Valid Total Inter-class Variance: {:.4f}, Valid Calinsky-Harabasz score: {:.4f}, '
           'Train Time: {:.4f}/epoch, Valid Time: {:.4f}/epoch')
    log_metrics = ([epoch]
                + [avg_metrics[split][m] for split in ['train', 'validate'] 
                   for m in ['loss', 'variances', 'ch_score']]
                + [v[-1] for v in model_time.values()])
    print(log.format(*log_metrics))
    

def define_log_path(log_metric_dir, config):
    """Define path to log file
    
    :param log_metrid_dir: pathlib. Directory of log file
    :param config: dict. Dictionary with configuration paramteres
    :return dataframe: log file with loss metrics
    """
    return (log_metric_dir 
            / (f"full_dataset_{config['full_dataset']}_"
               f"{config['loss']}_metrics_log.csv"))


def define_model(config, model_path=None):
    """Define ShadowScout model

    :param config: dict. Dictionary with model parameters
    :param model_path: str. Path to torch model file

    :return model: ShadowScout model
    :return device: cpu or gpu device
    """
    ### Model Initialization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ShadowScout(image_shape=config['img_shape'],
                        kernel_sigma=config['gaussian_kernel_sigma'], 
                        channel_weights=config['channel_weights'],
                        number_of_channels=config.get('number_of_channels', 3),
                        weights_clamp_range=config.get('channel_weights_clamp_range', 3),
                        learnable_weights=config['learnable_weights'],
                        dropout_rate=config.get('dropout_rate', 0.3))
    model.to(device)
    if model_path is not None:
        model.load_state_dict(model_path, map_location=device)
    model.to(device)

    return model, device

def set_seed(seed):
    torch.manual_seed(seed)  # Set seed for CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # Set seed for all GPUs
        torch.cuda.manual_seed_all(seed)  # If you are using multi-GPU
    np.random.seed(seed)  # Set seed for NumPy
    random.seed(seed)  # Set seed for the Python random module

    # Ensure deterministic behavior in convolutional layers
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(data, image_paths, config, save_dir):
    """Train ShadowNet model
    
    :param data: dict. Dictionary with dataset, stats and dataloader for train/val/test
    :param image_paths: dict. Dictionary with image paths for train/val/test
    :param config: dict. Dictionary with model parameters
    :param save_dir: pathlib. Directory where to save model to
    """
    
    ### Load the directory where the model checkpoints will be stored
    if save_dir.is_dir():
        rm_dir = input(f'Warning: {str(save_dir)} exists. Remove it? (y/n)')
        rm_dir = 'y'
        is_ok = 0
        while is_ok != 1:
            if rm_dir not in ['y', 'n']:
                print("Warning: please enter 'y' or 'n'")
                rm_dir = input(f'Warning: {str(save_dir)} exists. Remove it? (y/n)')
            else:
                is_ok = 1
        if rm_dir:
            shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # check channel orienation wrt shadows
    threshold_direction = determine_threshold_direction(
        data['train']['dataloader'], 
        len(image_paths['train']), 
        config.get('number_of_channels', 3))

    config['threshold_direction'] = threshold_direction
    
    seed = config.get('seed', np.random.choice(999))
    set_seed(seed)
    ### Model Initialization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if config.get('number_of_channels', 3) != len(config['channel_weights']):
        raise ValueError('number_of_channels must be same as len of channel_weights in config file')
    
    model, device = define_model(config)
    model.config = config
    model.config['stats'] = {k: [float(i) for i in v]
                             for k, v in data['train']['stats'].items()}
    
    ### Set optimizer and LR scheduler
    opt_network = [{'params': [param for name, param in model.named_parameters() if name != 'channel_weights.parameter'], 
                    'lr': config['learning_rate'], 
                    'weight_decay': config['weight_decay']}]
    if config['learnable_weights']:
        opt_network = opt_network + [{'params': [param for name, param in model.named_parameters() if name == 'channel_weights.parameter'], 
                                      'lr': config.get('learning_rate_weights', config['learning_rate']), 
                                      'weight_decay': config.get('weight_decay_weights', config['weight_decay'])}]
    optimizer = torch.optim.Adam(opt_network) 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5)

    channel_weight_alpha = list(config.get('channel_weight_alpha', 
                                           [1] * config['max_iters']))
    if len(channel_weight_alpha) != config['max_iters']:
        channel_weight_alpha = (channel_weight_alpha + [1] * config['max_iters'])[: config['max_iters']]
    
    ### Setup log file
    log_metric_file_path = define_log_path(save_dir, config)

    print('Running training ...')
    his_loss =[]
    model_time = {k: [] for k in ['train', 'validate']}
    min_loss = float('inf')
    wait = 0
    agg_functions = {m: np.nanmean if m != 'variances' else np.nansum
                     for m in ['loss', 'variances', 'ch_score']}
    ### Loop over epochs
    for epoch in progress_bar(range(config['max_iters'])):
        metrics = {i: {m: []
                   for m in agg_functions.keys()}
                   for i in ['train', 'validate']}
        keep_track = {k: 0 for k in ['train', 'validate']}
        
        ### adjust channel weights learning rate
        if config['learnable_weights']:
            optimizer.param_groups[1]['lr'] = config.get(
                'learning_rate_weights',
                config['learning_rate']) * channel_weight_alpha[epoch]
        
        
        ### Iterate over train and validate
        for split in keep_track.keys():
            t1 = time.time()
            if split== 'train':
                model.train()
            else:
                model.eval()
            ### Iterate over batches
            for idx, inputs in enumerate(data[split]['dataloader']):
                if model.learnable_weights:
                    channel_weights = np.array([float(w)
                                                for w in model.channel_weights.parameter])
                else:
                    channel_weights = np.array([float(w)
                                                for w in model.channel_weights])
                if split == 'train':
                    ### TRAIN
                    optimizer.zero_grad()
                    # for name, param in model.named_parameters():
                    #     if param.requires_grad is not None:
                    #         print(f"{name}: min={param.min().item()}, max={param.max().item()}, mean={param.mean().item()}")

                    normalized_inputs, thresholds, loss = make_predictions(
                        model, inputs, device)
                    if torch.isnan(loss):
                        print('Warning: loss value is nan, skipping train batch')
                        continue
                    metrics[split]['loss'].append(loss.cpu().detach() if torch.cuda.is_available() else loss.detach())
                    
                    loss.backward()
                    clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
                    optimizer.step()

                    # if channel_weights are learnable parameter, constraint it to config channel_weights_clamp_range
                    if model.learnable_weights:
                        model.channel_weights.constrain()

                    metrics[split], _, _ = collect_metrics(
                        normalized_inputs, 
                        thresholds, 
                        channel_weights,
                        metrics[split],
                        config)
                    keep_track[split] += thresholds.size(0)
                else:
                    ### VALIDATE
                    with torch.no_grad():
                        normalized_inputs, thresholds, loss = make_predictions(
                            model, inputs, device)
                        if torch.isnan(loss):
                            print('Warning: loss value is nan, skipping validation batch')
                            continue
                        metrics[split]['loss'].append(loss.cpu().detach() if torch.cuda.is_available() else loss.detach())
                        metrics[split], _, _ = collect_metrics(
                            normalized_inputs,
                            thresholds,
                            channel_weights,
                            metrics[split],
                            config)

                        keep_track[split] += thresholds.size(0)
            model_time[split].append(time.time()-t1)

        ### Aggregate losses
        avg_metrics = {split_key: {m_key: agg_functions[m_key](m_value)
                                   for m_key, m_value in metrics[split_key].items()}
                    for split_key in keep_track.keys()}
        his_loss.append(avg_metrics['validate']['loss'])
        
        ### Step with the learning rate scheduler
        scheduler.step(avg_metrics['validate']['loss'])

        ### Log epoch results
        if config['learnable_weights']:
            channel_weights = [float(c) for c in model.channel_weights.parameter]
        else:
            channel_weights = config['channel_weights']
                
        log_epoch(
            log_metric_file_path, epoch, avg_metrics, model_time, channel_weights)

        ### Check if model is current best and save
        if avg_metrics['validate']['loss'] < min_loss:
            model.save_state_dict(save_dir / f'best_epoch_checkpoint.pt')
            min_loss = avg_metrics['validate']['loss']
            print('>>>>>>>>>>>>>>>>>>>>>>>>> model saved <<<<<<<<<<<<<<<<<<<<<<<<<')
            wait = 0
        else:
            wait += 1
            if wait >= config['early_stop']:
                print('Early stopping.')
                break
    ### Identify best model and load
    bestid = np.argmin(his_loss)
    log = 'Best Valid Loss: {:.4f}'
    print(log.format(round(his_loss[bestid], 4)))

    # save learned channel weights
    if config['learnable_weights']:
        model, device = define_model(config, save_dir / f'best_epoch_checkpoint.pt')
        config['channel_weights_trained_model'] = [round(float(w), 2) for w in model.channel_weights.parameter]
    with open(save_dir / 'config.json', 'w') as config_file:
        json.dump(config, config_file)



def test_model(data, image_paths, config, save_dir, model_path, split='test', save_fname=''):
    """Run inference on ShadowNet model. Results are saved to json file containing thresholds per image
    
    :param data: dict. Dictionary with dataset, stats and dataloader for train/val/test
    :param image_paths: list. List with path to each image in train/val/test
    :param save_dir: pathlib. Directory where to save model to
    :param model_path: str. Path to torch model file
    :param split: str. On which dataset split to perform the test. Default='test'
    :param save_fname: str. Name of file to save results to (without the extension). Default='' (will be 'results_{split}')
    """

    model, device = define_model(config, model_path)
    config = model.config
    model.eval()
    results = {}
    keep_track = 0

    output_mask_dir = save_dir / 'predicted_masks'
    output_mask_dir.mkdir(parents=True, exist_ok=True)
    pb = progress_bar(list(zip(range(len(data[split]['dataloader'])), data[split]['dataloader'])))
    pb.comment = 'Running inference on batches...'
    for idx, inputs in pb:
        metrics = {i: {m: [] for m in ['loss', 'variances', 'ch_score']} for i in ['test']}
        with torch.no_grad():
            if model.learnable_weights:
                channel_weights = np.array([float(w) for w in model.channel_weights.parameter])
            else:
                channel_weights = np.array([float(w) for w in model.channel_weights])
                
            normalized_inputs, thresholds, loss = make_predictions(
                model, inputs, device)
            metrics['test']['loss'].append(loss.cpu().detach() if torch.cuda.is_available() else loss.detach())
            metrics['test'], detached_thresholds, output_masks = collect_metrics(
                normalized_inputs, 
                thresholds, 
                channel_weights, 
                metrics['test'],
                config)
            
            for i in range(thresholds.size(0)):
                results[image_paths[split][keep_track + i]] = {
                    'Interclass_variance': float(metrics['test']['variances'][i]),
                    'Calinski-Harabasz': float(metrics['test']['ch_score'][i]),
                    'H_I_threshold': float(detached_thresholds[i, 0]),
                    'I_threshold': float(detached_thresholds[i, 1]),
                    'S_threshold': float(detached_thresholds[i, 2])}
                for j in np.arange(3, config.get('number_of_channels', 3)):
                    results[image_paths[split][keep_track + i]].update({f'band{j+1}_threshold': float(detached_thresholds[i, j])})
                np.save(output_mask_dir / 
                        f'{Path(image_paths[split][keep_track + i]).stem}.npy',
                        np.uint8(output_masks[i]))
        keep_track += thresholds.size(0)

    results['stats'] = {k: tuple(float(v0) for v0 in v) for k, v in model.config['stats'].items()}
    if save_fname is not None:
        save_fname_ = f'results_{split}' if save_fname == '' else save_fname
        with open(save_dir / f'{save_fname_}.json', 'w') as f:
            json.dump(results, f)

    return model, results


if __name__ == "__main__":
    # ------------- Parse input arguments
    parser = argparse.ArgumentParser(description='Train Shadow Detection Model')
    parser.add_argument('mode', type=str, help='Specify run mode (train/infer)', choices=['train', 'infer'])
    parser.add_argument('--train_dir', type=str, required=False, help='Directory for training data')
    parser.add_argument('--validate_dir', type=str, required=False, help='Directory for validation data')
    parser.add_argument('--test_dir', type=str, required=False, help='Directory for testing data')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory for saving the models')
    parser.add_argument('--model_path', type=str, required=False, help='Path for model to use')
    parser.add_argument('--config_path', type=str, required=False, help='Path for model to use', default='src/shadowscout/config.json')

    args = parser.parse_args()

    # read config file
    config = json.load(open(args.config_path, 'rb'))
    # Prepare data for model
    image_paths, data = prepare_datasets(
        {k: getattr(args, f'{k}_dir')
         for k in ['train', 'validate', 'test']
         if getattr(args, f'{k}_dir') is not None},
        config)

    if args.mode == 'train':
        # Train model
        config['stats'] = data['train']['stats']
        train_model(data, image_paths, config, Path(args.save_dir))
    else:
        if not hasattr(args, 'model_path'):
            raise ValueError('If infering, --model_path argument must be passed')

        model, results = test_model(data, image_paths, config, Path(args.save_dir), args.model_path)

    

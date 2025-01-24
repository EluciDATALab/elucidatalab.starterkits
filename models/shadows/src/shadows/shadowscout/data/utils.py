import torch
import numpy as np
from fastprogress import progress_bar


def calculate_min_max_for_channels(dataset, number_of_channels=None):
    """
    Calculate the minimum and maximum values for different image channels across
    the dataset.
    if the dataset is too big (> 2100 images), only take 2100 random images to
    calculate the 95th percentile for the H-I channel.
    :param dataset: ShadowDataset, images dataset
    :param number_of_channels: int. Number of channels to calculate stats for.
    If None, use all available
    :returns: dictionary with min and max values (or min and 95th percentile
    for H_I) as tensors for the H-I, I and S channels (and NIR band if available).
    """
    number_of_channels = (len(dataset[0])
                          if number_of_channels is None
                          else number_of_channels)
    range_inputs = [[float('inf'), 0.0] for _ in range(number_of_channels)]
    
    if len(dataset) > 5000:
        randomly_chosen_indices = np.random.choice(range(len(dataset)), 
                                                   5000,
                                                   replace=False)
    else:
        randomly_chosen_indices = np.arange(len(dataset))
        
    # Loop through the dataset to collect all values
    pb = progress_bar(range(len(dataset)))
    pb.comment = 'Calculating channel range values...'
    for idx in pb:
        input_ = [i.detach().numpy()
                  if isinstance(i, torch.Tensor) else i
                  for i in dataset.__getitem__(idx)]
        range_inputs = [[np.min((range_inputs[k][0], np.min(i))),
                         np.max((range_inputs[k][1], np.max(i)))]
                        for k, i in enumerate(input_)]
        
    range_inputs[0][1] = dynamic_percentile([dataset[idx][0].detach().numpy()
                                             for idx in randomly_chosen_indices],
                                            *range_inputs[0],
                                            95)
    channels = ['H_I', 'I', 'S'] + [f'Band{i + 1}'
                                    for i in np.arange(3, number_of_channels)]

    stats = {key: tuple(torch.tensor(ri, dtype=torch.float64) for ri in range_input)
             for key, range_input in zip(channels, range_inputs)}
    return stats


def dynamic_percentile(datasets, min_value, max_value, ptile):
    """Dynamic percentile calculation for more efficient memory allocation
    
    :param datasets: ShadowDataset, images dataset
    :param min_value: float. Minimum HI value observed
    :param max_value: float. Maximum HI value observed
    :param ptile: int. Percentile to calculate
    
    :return percentile value
    """
    def calculate_histogram(datasets, min_value, max_value, bins=100):
        histogram = np.zeros(bins)
        total_count = 0
    
        # get histogram stats per dataset
        for dataset in datasets:
            array = dataset[0].flatten()
            hist, _ = np.histogram(array, bins=bins, range=(min_value, max_value))
            histogram += hist
            total_count += array.size
            del array  # Free memory
    
        return histogram, total_count, min_value, max_value
    
    def find_percentile(histogram, total_count, percentile, min_value, max_value):
        cumulative_histogram = np.cumsum(histogram)
        target_count = total_count * percentile / 100
        bin_width = (max_value - min_value) / len(histogram)
        bin_index = np.searchsorted(cumulative_histogram, target_count)
        bin_start = min_value + bin_index * bin_width
        bin_end = bin_start + bin_width
    
        return bin_start, bin_end
    
    def refine_percentile(datasets, bin_start, bin_end, percentile):
        values_in_bin = []
    
        for dataset in datasets:
            array = dataset[0].flatten()
            values_in_bin.extend(array[(array >= bin_start) & (array < bin_end)])
            del array  # Free memory
    
        values_in_bin = np.array(values_in_bin)
        return np.percentile(values_in_bin, percentile)

    # Step 1: First pass to collect histogram statistics
    histogram, total_count, min_value, max_value = calculate_histogram(datasets, min_value, max_value, bins=50)
    
    # Step 2: Find the approximate bin range where the percentile lies
    bin_start, bin_end = find_percentile(histogram, total_count, ptile, min_value, max_value)
    
    # Step 3: Refine the percentile calculation
    percentile_value = refine_percentile(datasets, bin_start, bin_end, ptile)

    return percentile_value

def normalize_channels(inputs, stats):
    """
    Normalizes the image channels based on provided statistics.
    :param inputs: list(Tensor), H-I, I and S channels plus optional NIR band
    :param stats: dict, dictionary with min and max values (or min and 95th percentile for H_I) as tensors for the H-I, I and S channels (and NIR band if available).
    return: the normalized inputs
    """
    
    normalized_inputs = [(inputs[i] - stats[key][0]) / (stats[key][1] - stats[key][0])
                         for i, key in enumerate(stats.keys()) if i < len(inputs)]
    normalized_inputs[0] = torch.clamp(normalized_inputs[0], min=0, max=1)

    return normalized_inputs


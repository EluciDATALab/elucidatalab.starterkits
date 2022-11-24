# ©, 2019, Sirris
# owner: GDN

"""
This file provides an implementation of the adaptive intelligent binning algorithm

This algorithm is described in De Meyer et al.'s "NMR-Based Characterization of Metabolic Alterations in Hypertension
Using an Adaptive, Intelligent Binning Algorithm". A summary of the paper can be found on the shares:
file:////server09.sirris.be/softeng/60_INTERNAL%20PROJECTS/EluciDATA%20Lab/5_Technical%20meetings/2017/20171214/
AIBA.pptx

Authors: GDN, TWT
"""


class AdaptiveIntelligentBinning(object):
    """
    This class implements the adaptive intelligent binning algorithm

    Usage: simply instantiate that class (see __init__), the result is stored in the "bin_indexes" instance variable

    Current limitations: the bin evaluation criterion does not currently take into account V_noise
    """

    def __init__(self, a_pandas_series, resolution):
        """
        Computes the binning on the provided Pandas series, with the provided resolution parameter

        :param a_pandas_series: a Pandas series from which bins need to be derived
        :param resolution: a real number providing the value of the resolution parameter
        """
        self.a_series = a_pandas_series
        self.resolution = resolution
        bin_value = self._compute_bin_value(0, a_pandas_series.size - 1)
        self.bin_indexes = self._adaptive_intelligent_binning_algorithm(bin_value, 0, a_pandas_series.size - 1)

    def _compute_bin_value(self, start_index, end_index):
        bin_value = 0
        for i in range(start_index, end_index + 1):
            sub_series_max = self.a_series.iloc[start_index:i+1].max()
            bin_value += ((sub_series_max - self.a_series.iloc[start_index]) *
                          (sub_series_max - self.a_series.iloc[i])) ** self.resolution

        return bin_value / (end_index - start_index + 1)

    def _adaptive_intelligent_binning_algorithm(self, bin_value, start_index, end_index):
        candidate = 0
        left_max_bin_value = 0
        right_max_bin_value = 0
        # we do not execute the loop for the last item in the series
        for index in range(start_index, end_index):
            left_bin_value = self._compute_bin_value(start_index, index)
            right_bin_value = self._compute_bin_value(index, end_index)
            if left_bin_value + right_bin_value > left_max_bin_value + right_max_bin_value:
                left_max_bin_value = left_bin_value
                right_max_bin_value = right_bin_value
                candidate = index

        # bin evaluation criterion
        if left_max_bin_value + right_max_bin_value > bin_value:
            return self._adaptive_intelligent_binning_algorithm(left_max_bin_value, start_index, candidate) + \
                   [candidate] + \
                   self._adaptive_intelligent_binning_algorithm(right_max_bin_value, candidate, end_index)
        else:
            return []

# ©, 2019, Sirris
# owner: PDAG

"""
A method implementing the relevancy score algorithm described in Dagnely, Pierre, Elena Tsiporkova, and Tom Tourwe

"Data-driven Relevancy Estimation for Event Logs Exploration and Preprocessing." ICAART (2). 2018.
A summary of the paper can be found on the shares

file:////server09.sirris.be/softeng/20_CONVENTIONS/Doctiris%202014/project%20execution/Dissemination/2017-08-05-ICAART2018/Data-driven%20Relevancy%20Estimation_for_Event_Logs_Exploration_and_Preprocessing.pdf
Authors: PDA
Usage: see test file for more details
"""

import pandas as pd
import numpy as np


def compute_relevancy_score(runs):
    """
    Compute the relevancy score of each events type of each run.

    For each run, the set of event is extracted and a score reflecting it's frequency in the run analyzed
    and in the other runs is associated to each event (type) of the set

    :param runs: A vector of vector. Each vector contains the list of events occurring during one run

    :returns A dataframe with the event type as index, each run is represented by a column (first run = first column),
        each cell contains the relevancy score of the corresponding event for the corresponding run
    """
    counter = 0
    df_term_frequency = pd.DataFrame()

    for run in runs:

        # compute tf
        dict_raw_frequency = {}
        count_events = pd.Series(run).value_counts()
        if len(count_events.index) > 0:
            for event in count_events.index:
                if event in dict_raw_frequency.keys():
                    dict_raw_frequency[event] = dict_raw_frequency[event] + count_events[event]
                else:
                    dict_raw_frequency[event] = count_events[event]

            df_raw_frequency = pd.DataFrame.from_dict(dict_raw_frequency, orient='index')
            df_raw_frequency_norm = df_raw_frequency / df_raw_frequency.max()
            df_raw_frequency_norm.columns = [str(counter)]
            counter += 1
            df_term_frequency = df_term_frequency.merge(df_raw_frequency_norm, how='outer', left_index=True,
                                                        right_index=True)

    # compute idf
    serie_term_frequency = pd.Series(np.log(len(df_term_frequency.columns) / (df_term_frequency.count(axis=1))))

    # compute tf_idf
    return df_term_frequency[[col for col in df_term_frequency.columns if 'idf' not in col]].fillna(0)\
        .multiply(serie_term_frequency, axis="index").dropna(axis=0)

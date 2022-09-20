# ©, 2022, Sirris
# owner: HCAB

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import random
import pandas as pd


def plot_grouped(df_original, df_compressed, group, signal, groups=None, random_drives=False):
    """Plot original vs. compressed signal, grouped by variable `group`."""
    df_original = df_original[[signal, group]]
    df_compressed = df_compressed[[signal, group]]
    
    if random_drives:
        random.seed(0)
        drives = df_compressed.drive.unique()
        groups = random.choices(drives, k=10)

    if not groups and random_drives:
        groups = list(set(df_original[group].unique())
                      & set(df_compressed[group].unique()))
        print('hello')

    ylim = [-0.2, 0.5]
    nrows = len(groups) // 2  # fix up if odd number of groups
    fig, axs = plt.subplots(figsize=(30, 5 * len(groups)),
                            nrows=nrows, ncols=2,  # fix as above
                            gridspec_kw=dict(hspace=0.4))  # Much control of gridspec
    for gr, ax in zip(groups, axs.flatten()):
        ax.plot(df_original[df_original[group] == gr][signal], label='original')
        ax.plot(df_compressed[df_compressed[group] == gr][signal], label='compressed', marker='P')
        ax.set_title(f'drive {gr}', fontsize=22)
        ax.set_ylabel(signal, fontsize=22)
        ax.set_xlabel('time', fontsize=22)
        ax.set_ylim(ylim)
        ax.legend()
    plt.show()


def plot_confusion_matrix(y_original, y_compressed_individually, y_compressed_combined, event_labels, original):
    # add in "No Event"s to build full confusion matrix
    y_original_ = pd.Series(y_original, index=original.index).fillna("No Event")  
    y_compressed_individually_ = pd.Series(y_compressed_individually, index=original.index).fillna("No Event")
    y_compressed_combined_ = pd.Series(y_compressed_combined, index=original.index).fillna("No Event")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 5))

    label_num = [2, 1, 517983, 10]
    cm_seq = confusion_matrix(y_original_, y_compressed_individually_)
    # need to add colorbar=False in plot when newer matplotlib library
    ConfusionMatrixDisplay((cm_seq/label_num)*100, display_labels=event_labels).plot(ax=ax1, values_format='.3g')
    ax1.set(title="Confusion matrix\noriginal vs. compressed individually")

    cm_comb = confusion_matrix(y_original_, y_compressed_combined_)
    ConfusionMatrixDisplay((cm_comb/label_num)*100, display_labels=event_labels).plot(ax=ax2, values_format='.3g')
    ax2.set(title="Confusion matrix\noriginal vs. compressed combined")
    fig.autofmt_xdate()

# ©, 2020, Sirris
# owner: RVEB

"""Collection of functions to perform adaptively refined binning"""

import ndtamr.NDTree as nd
import ndtamr.AMR as amr
import ndtamr.Vis as vis
import pandas as pd
from ndtamr.Data import GenericData
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt


def get_refined_cell_values(n):
    """
    Function to determine the number of points in the newly refined cells

    :param n: the node (cell) that needs to be refined
    :return: list containing the data for each new node (cell)
    """
    data = [None]*n.nchildren
    if n.data is not None:
        level = n.get_level(n.name)
        for i, c in enumerate(n.child):
            # Use func from CellData to get the values for the new cells.
            # Pass level + 1 to make sure you get the right number of points in the new cells
            data[i] = c._data_class(coords=c.coords, level=level+1)
    return data


def refinement_check(leaf, df, max_points=None, tol=None):
    """
    Our own function to determine whether a cell needs to be refined

    Based on the absolute number of points in the cell (if max_points is set)
    or on the fraction of points in the cell compared to the entire entire dataset (if tol is set)

    :param leaf: the tree leaf (cell) to check whether it needs to bee refined
    :param df: Pandas dataframe containing the data
    :param max_points: threshold for number of points in the cell before it is flagged for refinement. Default: None
    :param tol: threshold for fraction of points of the dataset in the cell before it is flagged for refinement.
        Default: None. Either max_points or tol needs to be set. If both are set, max_points is used.
    :return: whether it needs to be refined, the fraction of points of the dataset in the cell
    """
    assert tol is not None or max_points is not None, 'Pass either the maximum number of points or the tolerance'
    if max_points is not None:
        tol = max_points / df.shape[0]
    n_points = leaf.data.get_refinement_data()
    fraction = n_points / df.shape[0]

    ref = fraction >= tol

    leaf.rflag = ref

    return ref, fraction


def refine(tree, df, max_points=None, tol=None):
    """
    The main AMR routine which evaluates and refines each node in the tree.

    :param tree: The tree we want to refine
    :param df: Pandas DataFrame containing the data
    :param max_points: Refinement threshold for number of points in the cell
    :param tol: Refinement threshold for fraction of data points in the cell compared to all points in the dataframe
    :return: The total number of refined cells
    """
    depth = tree.depth()
    values = []
    for lvl in range(depth+1)[::-1]:
        tree.walk(target_level=lvl,
                  leaf_func=lambda x: values.append(refinement_check(x, df, tol=tol, max_points=max_points)[1]))
    total = amr.start_refine(tree)
    return total


def do_refinement(df_original, ref_vars, initial_bins_per_var, levels_of_refinement=0, max_points=1000):
    """
    The function provides a df containing the bin for each data point, the initial uniform tree, the refined tree

    :param df_original: Pandas DataFrame containing the data
    :param ref_vars: list of variables to do the binning on
    :param initial_bins_per_var: list of initial granularities
    :param levels_of_refinement: how many times to refine. If None, keep refining until each cell has less than
        max_points. Default: None
    :param max_points: Refinement threshold for number of points in the cell. Default: 1000
    :return: dataframe containing the bin for each data point, the initial uniform tree, the refined tree
    """
    assert len(ref_vars) == len(initial_bins_per_var)
    df = df_original.copy()
    scaler = MinMaxScaler()
    df[ref_vars] = scaler.fit_transform(df[ref_vars])
    dim = len(ref_vars)

    start_depth = int(np.ceil(np.log2(np.max(initial_bins_per_var))))
    xmins = tuple([0 for _ in ref_vars])
    xmaxs = tuple([2 ** start_depth / initials_bins for initials_bins in initial_bins_per_var])

    class CellData(GenericData):
        """Structure that handles the cell data"""

        default_coords = tuple([0 for _ in range(dim)])

        def __init__(self, coords=default_coords, file=None, data=None, level=start_depth):
            self.level = level
            GenericData.__init__(self, coords=coords, file=file, data=data)

        def func(self):
            """
            Function which determines how many points are within that cell

            The function sets the data value, gets the extent of the cell
            (using the starting coordinates and the level) and determines how many points are within that cell
            """
            x0s = self.coords
            x1s = [c + (xmax - xmin) * 2 ** -self.level for c, xmin, xmax in zip(self.coords, xmins, xmaxs)]

            subset_df = df[(df[ref_vars] >= x0s).all(axis=1) & (df[ref_vars] <= x1s).all(axis=1)]

            return max(1e-10, subset_df.shape[0])

        def get_refinement_data(self):
            """Returns the data column which we want to refine on."""
            return self.value

    # Make the initial grid
    t_uniform = nd.make_uniform(depth=start_depth,
                                dim=dim,
                                data_class=CellData,
                                xmin=xmins, xmax=xmaxs,
                                restrict_func=nd.restrict_average, prolongate_func=get_refined_cell_values)

    # Do the refinement
    t_refined = t_uniform.deepcopy()

    if levels_of_refinement is not None:
        for _ in range(levels_of_refinement):
            refine(t_refined, df, max_points=max_points)
            # amr.compression(t_refined)

    else:
        max_points_in_cell = np.max([leaf.data.get_refinement_data() for leaf in t_refined.list_leaves()])
        while max_points_in_cell > max_points:
            refine(t_refined, df, max_points=max_points)
            max_points_in_cell = np.max([leaf.data.get_refinement_data() for leaf in t_refined.list_leaves()])

    # Label the data points in the dataframe according to the bin they're in
    def label_bins(df_without_labels: pd.DataFrame, t):
        index_name = df_without_labels.index.name
        if index_name is None:
            index_name = 'index'
        df_with_labels = df_without_labels.copy().reset_index()
        for bin_id, leaf in enumerate(t.list_leaves()):
            coords = leaf.get_coords()
            level = leaf.get_level(leaf.name)

            x0s = coords
            x1s = [c + (xmax - xmin) * 2 ** -level for c, xmin, xmax in zip(coords, xmins, xmaxs)]

            subset_df = df_with_labels[(df_with_labels[ref_vars] >= x0s).all(axis=1) &
                                       (df_with_labels[ref_vars] <= x1s).all(axis=1)]
            # print(subset_df.index)
            df_with_labels.loc[subset_df.index, 'Bin'] = bin_id

        df_with_labels['Bin'] = df_with_labels['Bin'].astype(int)
        df_with_labels = df_with_labels.set_index(index_name)

        return df_with_labels

    df_with_bins = label_bins(df, t_refined)

    return df_with_bins, t_uniform, t_refined


def plot_2d(t, ref_vars, df, log=True):
    """
    Plot a 2D tree

    :param t: the tree
    :param ref_vars: list of variables
    :param df: dataframe containing the data
    :param log: whether to log-scale the number of points in the cells
    :return:
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
    vis.plot(t, grid=True, lognorm=log, fig=fig, ax=ax1, vmin=1, vmax=1e6, cb_kargs={'log': log}, cmap='Greys')
    vis.plot(t, grid=True, lognorm=log, fig=fig, ax=ax2, vmin=1, vmax=1e6, cb_kargs={'log': log}, cmap='RdBu')

    scaler = MinMaxScaler()
    df_scaled = df.copy()[ref_vars]
    df_scaled[ref_vars] = scaler.fit_transform(df_scaled)
    sns.scatterplot(data=df_scaled, x=ref_vars[0], y=ref_vars[1], ax=ax2, alpha=0.1, color='g')

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    return ax1, ax2


def plot_3d(t, ref_vars, df, log=True):
    """
    Plot a 3D tree

    :param t: the tree
    :param ref_vars: list of variables
    :param df: dataframe containing the data
    :param log: whether to log-scale the number of points in the cells
    :return:
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    # vis.plot(t, grid=True, lognorm=log, fig=fig, ax=ax1);
    # vis.plot(t, grid=True, lognorm=log, fig=fig, ax=ax2, vmin=1, vmax=1e6, cb_kargs={'log': log}, cmap='RdBu');

    vis.plot(t, grid=True, dims=[0, 1], slice_=[(-1, 0)], fig=fig, ax=axes[0, 0], vmin=1, vmax=1e6,
             cb_kargs={'log': log}, cmap='RdBu')
    vis.plot(t, grid=True, dims=[0, 2], slice_=[(1, 0)], fig=fig, ax=axes[0, 1], vmin=1, vmax=1e6,
             cb_kargs={'log': log}, cmap='RdBu')
    vis.plot(t, grid=True, dims=[1, 2], slice_=[(0, 0)], fig=fig, ax=axes[0, 2], vmin=1, vmax=1e6,
             cb_kargs={'log': log}, cmap='RdBu')

    vis.plot(t, grid=True, dims=[0, 1], slice_=[(-1, 0)], fig=fig, ax=axes[1, 0], vmin=1, vmax=1e6,
             cb_kargs={'log': log}, cmap='RdBu')
    vis.plot(t, grid=True, dims=[0, 2], slice_=[(1, 0)], fig=fig, ax=axes[1, 1], vmin=1, vmax=1e6,
             cb_kargs={'log': log}, cmap='RdBu')
    vis.plot(t, grid=True, dims=[1, 2], slice_=[(0, 0)], fig=fig, ax=axes[1, 2], vmin=1, vmax=1e6,
             cb_kargs={'log': log}, cmap='RdBu')

    scaler = MinMaxScaler()
    df_scaled = df.copy()[ref_vars]
    df_scaled[ref_vars] = scaler.fit_transform(df_scaled)
    sns.scatterplot(data=df_scaled, x=ref_vars[0], y=ref_vars[1], ax=axes[1, 0], alpha=0.1, color='g')
    sns.scatterplot(data=df_scaled, x=ref_vars[0], y=ref_vars[2], ax=axes[1, 1], alpha=0.1, color='g')
    sns.scatterplot(data=df_scaled, x=ref_vars[1], y=ref_vars[2], ax=axes[1, 2], alpha=0.1, color='g')

    for ax in axes.flatten():
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)


def get_cube_stats(df, columns):
    """
    Get some basic statistics for each cube

    :param df: dataframe containing the data
    :param columns: columns for which you want statistics
    :return: dictionary containing the summary statistics
    """
    assert 'Bin' in df.columns
    stats = {}
    for bin_number in df['Bin'].unique():
        df_bin = df[df['Bin'] == bin_number]
        stats_bin = {'median': df_bin[columns].median(), 'std': df_bin[columns].std(), 'min': df_bin[columns].min(),
                     'max': df_bin[columns].max()}
        stats[bin_number] = stats_bin
    return stats

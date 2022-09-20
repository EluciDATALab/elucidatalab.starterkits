# ©, 2019, Sirris
# owner: HCAB

"""Helper functions to visualize statistical summaries"""
import pandas as pd
import numpy as np
import seaborn as sb
from itertools import combinations
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from starterkits.visualization import vis_tools
from starterkits.feature_extraction.feature_utils \
    import pca_feature_importance


def plot_correlation_matrix(x,
                            axes=None,
                            label_high_corr=None,
                            kwargs=None,
                            plot=True):
    """
    Plot correlation matrix on a pandas dataframe.

    Based on
    https://seaborn.pydata.org/examples/many_pairwise_correlations.html.
    The function will only use numeric datatypes

    :param x : dataframe. Pandas dataframe to plot correlation matrix on. Can
    also be the calculated correlation matrix
    :param axes : axes object. Default: None
    :param label_high_corr : float. Threshold to use in labeling strongly
    correlated comparisons. Default : None (don't plot)
    :param kwargs : dict. Further parameters to be passed to seaborn.heatmap.
    :param plot : bool. Whether to plot correlation matrix. Default: True

    :returns x_corr: dataframe containing correlation matrix
    :returns df_corr: reshaped dataframe to columnar shape
    :returns axes: axes object
    """
    kwargs = {} if kwargs is None else kwargs

    sb.set(style="white")

    # check if input is correlation matrix already
    x_corr = None
    if x.shape[0] == x.shape[1]:
        if np.alltrue(x.index == x.columns):
            # Compute the correlation matrix
            x_corr = x

    if x_corr is None:
        x_corr = x.corr()

    if plot:

        # Generate a mask for the upper triangle
        mask = np.zeros_like(x_corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        # Set up the matplotlib figure
        if axes is None:
            f, axes = plt.subplots(figsize=(11, 9))

        # Generate a custom diverging colormap
        # cmap = sb.diverging_palette(220, 10, as_cmap=True)
        kwargs['cmap'] = kwargs.get('cmap', 'coolwarm')
        kwargs['center'] = kwargs.get('center', 0)

        # Draw the heatmap with the mask and correct aspect ratio
        sb.heatmap(x_corr, mask=mask, square=True,
                   cbar_kws={"shrink": .5}, ax=axes, **kwargs)
        axes.set_xticklabels(axes.get_xticklabels(),
                             rotation=45,
                             fontdict={'horizontalalignment': 'right'})

        axes.set_ylim([x_corr.shape[0], 0])
        axes.set_xlim([0, x_corr.shape[1]])

        if label_high_corr:
            # label correlations above threshold
            high_corr = np.triu(x_corr, k=1)
            high_corr = np.argwhere(high_corr > label_high_corr)

            for h in high_corr:
                axes.annotate('*', (h[0] + 0.25, h[1] + 0.75))
    else:
        axes = None

    df_corr = x_corr.values
    df_corr[np.tril_indices(df_corr.shape[0])] = np.nan
    df_corr = pd.DataFrame(df_corr, index=x_corr.index, columns=x_corr.columns)
    df_corr.index.name = df_corr.index.name or 'variable1'
    idx_name = df_corr.index.name
    df_corr = pd.melt(df_corr.reset_index(),
                      id_vars=idx_name,
                      var_name='variable2',
                      value_name='r').\
        rename(columns={idx_name: 'variable1'})
    df_corr = df_corr[df_corr.variable1 != df_corr.variable2]
    df_corr.dropna(inplace=True)
    df_corr['r_abs'] = np.abs(df_corr.r.abs())
    df_corr.sort_values('r_abs', ascending=False, inplace=True)

    return x_corr, df_corr, axes


def plot_pca_explained_variance(x=None,
                                pca=None,
                                n_comp=20,
                                ev_thresh=0.9,
                                ax=None):
    """
    Plot explained variance of PCA analysis

    :params: x : dataframe or array. Dataset to calculate PCAs on. Default :
    None.
    :params: pca : PCA object. PCA object. Default : None.
    :params: n_comp : int. Number of principal components to use in ev
    calculation. Default: 20
    :params: ev_thresh : float. Desired ratio of explained variance.
    Default : 0.9
    :params: ax: axes object. Default: None
    Returns
    ------
    """
    if pca is None:
        # get explained variance of PCA analysis
        pca = PCA(n_components=n_comp).fit(x)
    ev_c = np.cumsum(pca.explained_variance_ratio_)

    # plot
    if ax is None:
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
    ax.plot(np.arange(len(ev_c)), ev_c, 'o-')

    # define minimum number of PCAs to explain ev_thresh % of variance
    n_comp_target = np.argwhere(ev_c > ev_thresh)
    n_comp_target = n_comp_target.min() + 1 if len(n_comp_target) > 0 else None
    if n_comp_target is not None:
        ax.axvline(n_comp_target - 1, linestyle='--', color='red')
        ax.axhline(ev_c[np.argwhere(ev_c > ev_thresh).min()],
                   linestyle='--',
                   color='red')
        ax.set_title('%d PCA components explain %0.1f%% of variance in the '
                     'dataset' %
                     (n_comp_target, ev_c[n_comp_target].min() * 100))
    else:
        ax.set_title('Explained variance of PCA analysis.\nFailed to reach '
                     'desired threshold')
    ax.set_xlabel('PCA component')
    ax.set_xticks(np.arange(n_comp))
    ax.set_xticklabels(np.arange(n_comp) + 1)
    ax.set_ylabel('Cumulative explained variance')

    return n_comp_target, pca


def plot_pca_feature_importance(feature_names,
                                x=None,
                                pca=None,
                                n_comp=20,
                                scale=False,
                                scale_importance=True,
                                thresh=None,
                                type_thresh='absolute',
                                plot=True):
    """
    Plot PCA derived feature importance

    :param x : dataframe or array. Dataset to calculate PCAs on. Default : None
    :param pca : PCA object. PCA object. Default : None.
    :param feature_names : list. List of feature names
    :param n_comp : int. Number of PCA components to extract (if pca is None).
    Default: 20
    :param scale: bool. Whether to scale the values in the dataset using
    StandardScaler. Default: False
    :param scale_importance : bool. Should importance values add to 1?
    Default : True.
    :param thresh : int. Percentile value to highlight features to exclude.
    Default :None.
    :param type_thresh : str. Type of thresholding to apply. Can be 'absolute'
    (use the value of thresh) or 'percentile' (take the thresh-th percentile).
    Default: 'absolute'
    :param plot: bool. Whether to display the plot. default: True

    :returns feat_importance: dataframe with feature importance of each feature
    :returns axes: axes object
    """
    feat_importance = pca_feature_importance(feature_names,
                                             x=x,
                                             pca=pca,
                                             n_comp=n_comp,
                                             scale=scale,
                                             scale_importance=scale_importance)

    if thresh is not None:
        if type_thresh == 'percentile':
            thresh = np.percentile(feat_importance.importance, thresh)
        feat_importance['exclude'] = feat_importance.importance < thresh
    else:
        feat_importance['exclude'] = False

    if plot:
        fig, axes = plt.subplots(figsize=(15, 5))
        cols = vis_tools.define_color_categories([True, False], cmap='Set1')
        for c in [False, True]:
            fi = feat_importance[feat_importance.exclude == c]
            axes.bar(fi.feature, fi.importance, color=cols[c])
        if feat_importance.exclude.nunique() > 1:
            axes.legend(handles=vis_tools.define_color_patches(cols),
                        title='Exclude')
        if thresh is not None:
            axes.axhline(thresh, linestyle='--', color=cols[True])
        axes = vis_tools.format_axes(axes, rotate=45, is_date=False)
        axes.set_title('Feature importance')
        axes.set_ylabel('Importance')
        axes.set_xlabel('Feature')
    else:
        axes = None

    return feat_importance, axes


def plot_cluster_identify(x,
                          cluster_col='cluster',
                          use_features='first3',
                          n_comb=3):
    """
    Plot the cluster each sample was assigned to using different combinations

    of variables. It also supports deriving PCAs and using those for
    visualization

    :params: x : dataframe. Dataframe with features and cluster ID
    :params: cluster_col : str. Name of cluster ID column. Default : 'cluster'
    :params: use_features : str. Which features to use. 'First3' will use the
    first 3 columns in the dataset. 'PCA' will derive principal components and
    use the first 3. Can also be a list specifying the names of the columns to
    use. Default: 'first3'
    :params: n_comb: int. Number of combinations to visualize. Default : 3
    Returns
    ------
    """
    if isinstance(use_features, list):
        pass
    elif use_features.lower() == 'pca':
        x_pca = PCA(n_components=3).fit_transform(x.drop(cluster_col, axis=1))
        clusters = x[cluster_col].values
        x = pd.DataFrame(x_pca, columns=['PCA1', 'PCA2', 'PCA3'])
        x[cluster_col] = clusters
        use_features = ['PCA1', 'PCA2', 'PCA3']
    elif use_features.lower() == 'first3':
        use_features = x.columns[:3]
    else:
        raise ValueError('Provide a valid input for use_features')

    combs = list(combinations(use_features, 2))
    nrow, ncol = vis_tools.optimal_layout(n_comb)
    fig, axes = plt.subplots(nrow, ncol, figsize=(7 * ncol, 7 * nrow))
    axes = [axes] if n_comb == 1 else axes.flatten()
    cols = vis_tools.define_color_categories(x[cluster_col].unique())
    ct = -1
    for k, c in enumerate(combs[:n_comb]):
        sb.scatterplot(x=c[0], y=c[1], hue=cluster_col, palette=cols, data=x,
                       ax=axes[k], s=200, alpha=0.5)
        ct += 1
    vis_tools.clean_remaining(axes, ct)

    return fig, axes


def plot_scatter_pearson(df, var1, var2, group=None, facet_format='2:3',
                         figsize=(14, 7), limit=10000):
    """
    Plot scatterplot between two variables with a linear fit and the pearson

    correlation value in the title

    :params: df : dataframe. Dataframe
    :params: var1 : str. Name of variable in the x-axis
    :params: var2 : str. Name of variable in the y-axis
    :params: group : str. Name of variable to group dataset by. Default : None
    :params: facet_format : str. Format of figure layout for grouped datasets.
    See vis_tools.optimal_layout for more info. Default : '2:3'
    :params: figsize : tuple. Tuple with figure size. Default : (14, 7)
    :params: limit : int. Maximum number of samples to use (taken randomly).
    Default: 10000
    Returns
    ------
    """
    df_nonnull = df.dropna()
    if len(df_nonnull) > limit:
        df_nonnull = df_nonnull.sample(limit)

    def make(x, y, ax, add_title=''):
        # make linear fit
        f = np.poly1d(np.polyfit(x, y, 1))
        xvec = np.linspace(x.min(), x.max(), 100)
        yvec = f(xvec)
        # get pearson correlation
        r, p = pearsonr(x, y)
        sb.scatterplot(x=x, y=y, alpha=0.4, ax=axes)
        ax.plot(xvec, yvec, color='red')
        ax.set_title('%sPearson R=%0.2f,\npval=%0.2e' % (add_title, r, p))

    if group is None:
        # plot
        fig, axes = plt.subplots(figsize=figsize)
        make(df_nonnull[var1], df_nonnull[var2], axes)
    else:
        ncol, nrow = vis_tools.optimal_layout(df[group].nunique(),
                                              layout_format=facet_format)
        fig, axes = plt.subplots(ncol, nrow, figsize=figsize)
        axes = axes.flatten()
        ct = -1
        for k, g in enumerate(df_nonnull[group].unique()):
            make(df_nonnull[df_nonnull[group] == g].dropna()[var1],
                 df_nonnull[df_nonnull[group] == g].dropna()[var2],
                 axes[k],
                 add_title='%s: ' % g)
            ct += 1
        fig.tight_layout()
        vis_tools.clean_remaining(axes, ct)

    return axes


def plot_cumulative_curve(x, target,
                          group=None,
                          figsize=(14, 7),
                          plot_type='percentage',
                          x_scale_log=False):
    """
    Plot the cumulative curve of a target variable

    :params: x : dataframe. Dataframe
    :params: target : str. Name of variable to plot
    :params: group : str. Name of variable to group dataset by. Default : None
    :params: figsize : tuple. Tuple with figure size. Default : (14, 7)
    :params: plot_type : str. Type of cumulative plot. Options are 'percentage'
    and 'absolute'. Default : percentage
    :params: x_scale_log : bool. Use logarithmic x-scale? Default : False

    :returns x_count : dataframe. Dataframe with cumulative statistics
    :returns axes : axes object
    """
    if group is None:
        x['dummy_grouper'] = 1
        group = 'dummy_grouper'
    x_count = x.groupby([target, group]).size().reset_index(name='ct')

    x_count['cum_ct'] = x_count.groupby(group).ct.cumsum()
    x_count['percent'] = x_count.groupby(group).\
        ct.\
        apply(lambda xx: xx / xx.sum() * 100)
    x_count['cum_percent'] = x_count.groupby(group).percent.cumsum()

    plot_type = 'cum_percent' if plot_type == 'percentage' else 'cum_ct'
    group_cols = vis_tools.define_color_categories(x_count[group].unique())
    fig, axes = plt.subplots(figsize=figsize)
    for g in group_cols:
        x_count0 = x_count[x_count[group] == g]
        axes.plot(x_count0[target], x_count0[plot_type],
                  '-o',
                  color=group_cols[g])

    if x_scale_log is True:
        axes.set_xscale('log')

    if group != 'dummy_grouper':
        axes.legend(handles=vis_tools.define_color_patches(group_cols))
    else:
        x.drop('dummy_grouper', axis=1)

    return x_count, axes


def plot_cluster_feature_distribution(ds,
                                      cluster_col='cluster',
                                      remove_extreme=True,
                                      cluster_colors=None,
                                      fig_size=None):
    """
    Plot distribution of values per cluster

    :params: ds : dataframe. Dataframe
    :params: cluster_col : str. Name of variable with cluster ID
    :params: remove_extreme : bool. Whether to remove values not between 1st
    and 99th
    percentile for numerical variables. Default : True
    :params: cluster_colors : dict. Dictionary with colors for each cluster.
    Default: None.
    :params: fig_size : tuple. Matplotlib figure size. Default : None
    Returns
    ------
    """
    # prep dataset
    ds.copy()
    # define figure layout
    ncol, nrow = vis_tools.optimal_layout(ds.shape[1])
    fig, axes = plt.subplots(ncol,
                             nrow,
                             figsize=fig_size or (ncol * 5, nrow * 3))
    axes = axes.flatten() if ds.shape[1] > 1 else [axes]
    if cluster_colors is None:
        cluster_colors = \
            vis_tools.define_color_categories(sorted(ds.index.unique()))
    ct = -1
    for k, f in enumerate(ds.columns):
        if pd.api.types.is_numeric_dtype(ds[f]):
            # remove extreme values
            if remove_extreme:
                ds_f = ds[ds[f].between(*np.percentile(ds[f], [1, 99]))]
            else:
                ds_f = ds.copy()
            # plot distribution numeric values
            for c in cluster_colors:
                sb.distplot(ds_f.loc[c, f],
                            ax=axes[k],
                            color=list(cluster_colors[c]),
                            hist=False)
                # axes[k].legend().remove()
                axes[k].legend(
                    handles=vis_tools.define_color_patches(cluster_colors),
                    title='Cluster')
        else:
            # get percentage per cluster and class and plot
            ds_count = ds.groupby([f, pd.Grouper(level=cluster_col)]).\
                size().\
                reset_index(name='ct')
            ds_count['percent'] = ds_count.groupby(cluster_col).\
                ct.\
                apply(lambda x: x / x.sum() * 100)
            f_col = vis_tools.define_color_categories(ds_count[f].unique(),
                                                      'Set1')
            all_cluster = pd.DataFrame(
                {cluster_col: ds_count[cluster_col].unique(),
                 'lowest': np.zeros(ds_count[cluster_col].nunique())})
            for _, c in enumerate(ds_count[f].unique()):
                ds_count0 = ds_count[ds_count[f] == c].merge(all_cluster,
                                                             on=cluster_col,
                                                             how='right')
                ds_count0['percent'].fillna(0, inplace=True)
                axes[k].bar(ds_count0[cluster_col], ds_count0.percent,
                            bottom=ds_count0.lowest, color=f_col[c])
                all_cluster['lowest'] = all_cluster.lowest + ds_count0.percent
            # sb.barplot(x=cluster_col, y='percent', hue=f, data=ds_count,
            #            palette=f_col, ax=axes[k])
            axes[k].legend(handles=vis_tools.define_color_patches(f_col),
                           loc='upper center', bbox_to_anchor=(0.5, -0.05))
            axes[k].set_xticks(np.arange(ds_count[cluster_col].nunique()))
            axes[k].set_xticklabels(sorted(ds_count[cluster_col].unique()))
        axes[k].set_title(f)
        axes[k].set_xlabel('')
        axes[k].set_ylabel('')
        ct += 1
    fig.tight_layout()
    vis_tools.clean_remaining(axes, ct)

import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
from tqdm.notebook import tqdm

from starterkits.visualization import vis_tools

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score



def plot_dtw_convergence_radius_tuning(results):
    results['z_dtw'] = (results
                        .groupby('loop')
                        .dtw
                        .transform(lambda x: (x - x.mean()) / x.std()))
    plt.figure(figsize=(18, 6))
    results_sum = (results
                   .groupby('step')
                   .z_dtw
                   .agg([np.mean, np.std])
                   .reset_index())
    plt.plot(results_sum.step, results_sum['mean'], color=sb.xkcd_rgb['cool blue'])
    plt.scatter(results_sum.step, results_sum['mean'], color=sb.xkcd_rgb['cool blue'])
    vis_tools.axes_font_size('Radius', 
                             'Normalized DTW distance', 
                             'Convergence of DTW distance')


def get_radius_at_convergion(results, thresh_0=0):
    results['dt'] = results.groupby(['LCLid', 'date', 'loop'])['dtw'].diff()
    results.loc[~(results.dt < thresh_0), 'dt'] = np.nan
    results['rk'] = (results
                     .groupby(['LCLid', 'date', 'loop'])['dt']
                     .rank(method='first', ascending=False))

    plt.boxplot(results[results.rk == 1].step, vert=False)
    thresh = np.percentile(results[results.rk == 1].step, 95)
    plt.axvline(thresh, color='red')
    print('%0.1f%% of all comparisons converge within %d steps' % (95, thresh))

    return thresh


def get_best_number_clusters(dist_matrix):

    X = dist_matrix.values

    clu_vec = np.arange(2, 10)
    score = np.empty((2, len(clu_vec)))
    for k, c in tqdm(enumerate(clu_vec), total=len(clu_vec)):
        clu = AgglomerativeClustering(n_clusters=c, affinity='precomputed', linkage='complete').fit(X)
        clu_labels = clu.labels_
        for k2, m in enumerate([calinski_harabasz_score, davies_bouldin_score]):
            score[k2, k] = m(X, clu_labels)

    fig, axes = plt.subplots(figsize=(15, 5))
    m_cols = vis_tools.define_color_categories(['calinski_harabasz_score', 'davies_bouldin_score'], 'Set1')
    axes.plot(clu_vec, score[0, :], 'o-', 
            markersize=20,
            label='calinski_harabasz_score', 
            color=m_cols['calinski_harabasz_score'])
    ax2 = vis_tools.set_twin_yaxis(axes, label='davies_bouldin_score', ax_color=m_cols['davies_bouldin_score'])
    ax2.plot(clu_vec, score[1, :], 'o-', 
            markersize=20,
            label='davies_bouldin_score', 
            color=m_cols['davies_bouldin_score'])
    axes.legend(handles=vis_tools.define_color_patches(m_cols), loc='center right')
    vis_tools.axes_font_size(axes, 'Number of clusters', 'calinski_harabasz_score', '')
    vis_tools.axes_font_size(ax2, 'Number of clusters', 'davies_bouldin_score', '')

    return X
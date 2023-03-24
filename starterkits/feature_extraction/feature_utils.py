# ©, 2019, Sirris
# owner: HCAB

"""Tools to visualize, ... the importance of features"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def pca_feature_importance(feature_names, x=None, pca=None, scale=False, n_comp=3, scale_importance=True):
    """
    Derive feature importance from PCA analysis using the variance of each PCA and the PCA value of each feature

    :param feature_names : list. List of feature names
    :param x : dataframe or array. Dataset to calculate PCAs on. Default : None.
    :param pca : PCA object. PCA object. Default : None.
    :param scale: bool. Whether to scale the values in the dataset using StandardScaler. Default: False
    :param n_comp : int. Number of PCA components to extract (if pca is None). Default : 20
    :param scale_importance : bool. Should importance values add to 1? Default : True.

    :returns dataframe with feature importance of each feature
    """
    if (x is None) & (pca is None):
        raise ValueError('Either x or pca need to be provided')

    if (x is not None) & (isinstance(x, pd.DataFrame)):
        x = x.values

    if scale & (x is not None):
        x = StandardScaler().fit_transform(x)

    if pca is None:
        pca = PCA(n_components=n_comp).fit(x)

    pca_comp = pca.components_
    pca_ev = pca.explained_variance_ratio_

    # normalize feature PCA values of each feature per PCA
    pca_comp = np.apply_along_axis(lambda x: x / x.sum(), 1, np.abs(pca_comp))

    # multiply by explained variance of each PCA
    feat_importance = np.apply_along_axis(lambda x: x * pca_ev, 0, pca_comp)

    feat_importance = pd.DataFrame({'feature': feature_names,
                                    'importance': feat_importance.sum(axis=0)})
    feat_importance.sort_values('importance', ascending=False, inplace=True)

    if scale_importance:
        feat_importance['importance'] = feat_importance['importance'] / \
                                        feat_importance['importance'].sum()

    return feat_importance

import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from dtwalign import dtw
from itertools import combinations
from tslearn.metrics import dtw_path
from elucidata.tools.clustering import kmedoids
from starterkits.visualization import vis_tools
from scipy.spatial.distance import squareform



def get_holidays(ds, data_dir):
    holidays = pd.read_csv((data_dir / 'uk_bank_holidays.csv'))
    holidays.rename(columns={'Bank holidays': 'day'}, inplace=True)
    holidays['day'] = pd.to_datetime(holidays['day'])
    holidays['Type'] = 'bank_holiday'


    school_holidays = pd.read_csv((data_dir / 'term dates.csv'))
    school_holidays['date'] = pd.to_datetime(school_holidays['date'], format='%d/%m/%Y')
    school_holidays.sort_values('date', inplace=True)

    school_holidays_expand = []
    for k, r in school_holidays[school_holidays.schoolYear > '2010'].iterrows():
        if r.schoolStatus == 'Close':
            school_holidays_expand.append(pd.DataFrame({'day': pd.date_range(r.date, school_holidays.loc[k+1, 'date'], freq='D'),
                    'school_holiday': 'school holiday'}))
    school_holidays_expand = pd.concat(school_holidays_expand)

    holidays = holidays.merge(school_holidays_expand, on='day', how='outer')
    holidays['day_type'] = holidays.Type.combine_first(holidays.school_holiday)

    # merge all
    ds = (ds
          .reset_index()
          .merge(holidays.rename(columns={'day': 'datetime'})[['datetime', 'day_type']], 
          on='datetime', 
          how='left')
          .set_index(['LCLid', 'datetime']))

    ds['day_type'].fillna('normal', inplace=True)

    return ds


def finetune_dtw_radius(ds, cols):
    cols = [cols] if isinstance(cols, str) else cols
    np.random.seed(333)
    pool = (ds
            .loc[ds.day_type == 'normal']
            .reset_index()
            [['LCLid', 'date', 'wday', 'quarter']]
            .drop_duplicates())
    
    rnd_pool = pool.sample(100, replace=False)

    steps = np.arange(15)
    out = {'LCLid': [], 'date': [], 'wday': [], 'loop': [], 'step': [], 'dtw': []}
    for _, r in tqdm(rnd_pool.iterrows(), total=len(rnd_pool)):
        pool_options = (pool[(pool.wday == r.wday) 
                             & (pool['date'] != r['date'])
                             & (pool.LCLid == r.LCLid)
                             & (pool.quarter == r.quarter)])
        if len(pool_options) > 10:
            pool_options = pool_options.sample(10, replace=False).reset_index(drop=True)
        array1 = ds.loc[pd.IndexSlice[r.LCLid,str(r.date)], cols].values
        for k2, r2 in pool_options.iterrows():
            array2 = ds.loc[pd.IndexSlice[r.LCLid,str(r2.date)], cols].values
            for s in steps:
                p, d = dtw_path(array1, 
                                array2, 
                                global_constraint='sakoe_chiba',
                                sakoe_chiba_radius=s)
                out['wday'].append(r.wday)
                out['LCLid'].append(r.LCLid)
                out['date'].append(r.date)
                out['step'].append(s)
                out['loop'].append(k2)
                out['dtw'].append(d / len(p))
    out = pd.DataFrame(out)

    return out


def get_distance_matrix(ds, radius, cols):
    cols = [cols] if isinstance(cols, str) else cols
    ds_units = ds.reset_index()[['LCLid', 'date']].drop_duplicates().reset_index(drop=True)
    pairwise = list(combinations(ds_units.index, 2))
    ds_dist_matrix = {'idx1': [], 'idx2': [], 'dtw': []}
    for k in tqdm(pairwise, total=len(pairwise)):
        r1 = ds_units.loc[k[0]]
        r2 = ds_units.loc[k[1]]
        array1 = ds.loc[(ds.index.get_level_values(0) == r1.LCLid) 
                         & (ds['date'] == r1.date), cols].values
        array2 = ds.loc[(ds.index.get_level_values(0) == r2.LCLid) 
                        & (ds['date'] == r2.date), cols].values

        path, d = dtw_path(array1, 
                        array2, 
                        global_constraint='sakoe_chiba',
                        sakoe_chiba_radius=radius)
        ds_dist_matrix['idx1'].append(k[0])
        ds_dist_matrix['idx2'].append(k[1])
        ds_dist_matrix['dtw'].append(d / len(path))
        
    ds_dist_matrix = pd.DataFrame(ds_dist_matrix)

    dist_matrix = pd.DataFrame(squareform(ds_dist_matrix.dtw),
                                  index=ds_units.index,
                                  columns=ds_units.index)

    return dist_matrix, ds_dist_matrix, ds_units


def make_kmedoid_clustering(dist_matrix, ds_units, n_clusters):
    iseed=979
    np.random.seed(iseed)
    kmed = kmedoids.KMedoids(n_clusters, verbose=False, number_of_iterations=1)
    kmed.fit(dist_matrix.values, kwargs={'data_type':'distance_matrix'})
    
    clu_labels = kmed.best_cluster_labels
    clu_day = ds_units.copy()
    clu_day['cluster'] = np.int64(clu_labels)

    clu_lclid = clu_day.groupby(['LCLid', 'cluster']).size().reset_index(name='ct')
    clu_lclid['rk'] = clu_lclid.groupby('LCLid').ct.rank(ascending=False, method='first')
    clu_lclid['percent'] = clu_lclid.groupby('LCLid').ct.transform(lambda x: x / x.sum() * 100)
    
    clu_cols = vis_tools.define_color_categories(clu_lclid.cluster.unique())  

    return clu_day, clu_lclid, clu_cols


def add_cluster_to_results(ds_dist_matrix, cluster_day):
    ds_dist_matrix.drop(columns=['cluster1', 'cluster2'], errors='ignore', inplace=True)
    ds_dist_matrix = (ds_dist_matrix
                      .merge(cluster_day[['cluster']]
                             .reset_index()
                             .rename(columns={'index': 'idx1', 
                                              'cluster': 'cluster1'}), 
                             on=['idx1'], 
                             how='left')
                      .merge(cluster_day[['cluster']]
                             .reset_index()
                             .rename(columns={'index': 'idx2', 
                                              'cluster': 'cluster2'}), 
                             on=['idx2'], 
                             how='left'))

    cluster_ranks = (ds_dist_matrix
                     .loc[ds_dist_matrix.cluster1==ds_dist_matrix.cluster2,
                          ['idx1', 'cluster1', 'dtw']]
                      .rename(columns={'idx1': 'index', 
                                       'cluster1': 'cluster'}))

    cluster_ranks['rks'] = cluster_ranks.groupby('cluster').dtw.rank(method='first')
    cluster_centers = (cluster_ranks
                       .loc[cluster_ranks.rks == 1, ['index', 'cluster']]
                       .set_index('cluster'))

    return ds_dist_matrix, cluster_ranks, cluster_centers
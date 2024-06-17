import plotly.express as px
import pandas as pd

from sklearn.metrics import root_mean_squared_error


def show_clusters(fedrepo):
    """ Plot evaluation vectors of all clients, color coded by cluster label. """
    df = fedrepo.evaluation_vectors.T.melt(ignore_index=False).reset_index().set_index('client_id')
    df['label'] = -1

    for i, clus in enumerate(fedrepo.clusters[-1]):
        df.loc[clus, 'label'] = i

    fig = px.line(df.reset_index().sort_values(fedrepo.evaluation_vectors.columns.name),
                  y='value',
                  color='label',
                  x=fedrepo.evaluation_vectors.columns.name,
                  line_group='client_id',
                  template='plotly_white',
                  labels={'tree_num': 'Trees in global model', 'label': 'Cluster'}
                  )
    fig.show()


def show_static_performance(fedrepo):
    """ Plot RMSE scores of global, local and (static) cluster models. """
    d = {worker: [root_mean_squared_error(pred_dict['true'], pred_dict['pred_local']),
                  root_mean_squared_error(pred_dict['true'], pred_dict['pred_cluster']),
                  root_mean_squared_error(pred_dict['true'], pred_dict['pred_global'])
                  ] for worker, pred_dict in fedrepo.without_maintenance_dict.items()}

    df = pd.DataFrame(d, index=['local', 'cluster', 'global']).T
    for i, clus in enumerate(fedrepo.clusters[-1]):
        df.loc[clus, 'label'] = str(i)

    df = df.drop('MAC004863')
    df = df.melt(id_vars='label', ignore_index=False).sort_values(['label', 'variable'])

    # Calculate the number of unique clusters to determine the height
    num_clusters = len(df['label'].unique())
    rows_needed = (num_clusters + 2) // 3  
    height = 300 * rows_needed 

    fig = px.line(df,
                  y='value',
                  color='label',
                  template='plotly_white',
                  facet_col='label',
                  facet_col_wrap=3,
                  line_dash='variable',
                  line_dash_sequence=['dot', 'solid', 'dash'],
                  labels={'variable': 'Model', 'value': 'RMSE', 'index': 'Workers'},
                  height=height,
                  )
    fig.update_xaxes(matches=None, tickangle=45, showticklabels=False)
    fig.update_yaxes(matches=None)
    fig.for_each_annotation(lambda a: a.update(text=f'Cluster {int(a.text.split("=")[-1]) + 1}'))

    labels_seen = set()
    fig.for_each_trace(lambda trace_: trace_.update(showlegend=False) if (trace_.name[3:] in labels_seen)
                       else labels_seen.add(trace_.name[3:]))

    for trace in fig.data:
        trace.name = trace.name.split(", ")[1]
    fig.update_layout(legend_title_text='Model')

    fig.update_traces(line=dict(width=1.5))

    fig.show()


def show_worker_repo_hist(fedrepo):
    """ Plot the amount of workers in the workers' repo throughout the validation data. """
    fig = px.line(x=pd.date_range(start='2012-08-01', end='2013-12-31', freq='D'), y=fedrepo.worker_repository_hist, template='plotly_white', title="Workers present in workers' repo", labels={'y': 'Count', 'x': ''})
    fig.show()


def show_maintenance_example(fedrepo):
    """ Plot maintenance effect on daily RMSE for one example worker. """
    worker = 'MAC004556'
    df = pd.DataFrame(fedrepo.without_maintenance_dict[worker]).loc[fedrepo.with_maintenance_dict[worker]['pred_index']]
    df['pred_dynamic'] = fedrepo.with_maintenance_dict[worker]['predictions']


    def calculate_rmse(group):
        return pd.Series({'rmse_dynamic': root_mean_squared_error(group['true'], group['pred_dynamic']), 'rmse_cluster': root_mean_squared_error(group['true'], group['pred_cluster'])})

    rmse_daily = df.groupby(df.index.date).apply(calculate_rmse)
    retraining = [timestamp for timestamp, names in fedrepo.maintenance_log.items() if worker in names]
    
    fig = px.line(rmse_daily, template='plotly_white', labels={'value': 'RMSE', 'index': ''}, title='Daily RMSE scores for worker MAC004556')
    for time in retraining:
        fig.add_vline(x=time)
    fig.show()


def show_maintenance_effect(fedrepo):
    """ Plot prediction improvements of all workers involved in maintenance. """
    influenced = []
    for workers in fedrepo.maintenance_log.values():
        influenced.extend(workers)
        
    influenced = list(set(influenced))

    rmse_compare = pd.DataFrame()
    for worker in influenced:
        df = pd.DataFrame(fedrepo.without_maintenance_dict[worker]).loc[fedrepo.with_maintenance_dict[worker]['pred_index']]
        df['pred_dynamic'] = fedrepo.with_maintenance_dict[worker]['predictions']
        df['res_static'] = (df['pred_cluster'] - df['true']).abs()
        df['res_dynamic'] = (df['pred_dynamic'] - df['true']).abs()
        
        df['res_diff'] = df['res_static'] - df['res_dynamic']    
        rmse_compare.at[worker, 'median'] = df['res_diff'].mean()

    fig = px.bar(rmse_compare.sort_values('median'), template='plotly_white', labels={'value': 'Prediction improvement', 'index': ''})
    fig.update_layout(showlegend=False)
    fig.show()
    

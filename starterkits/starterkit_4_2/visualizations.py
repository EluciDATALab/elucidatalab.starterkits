import plotly.express as px


def show_clusters(evaluation_vectors, clusters):
    """ Plot evaluation vectors of all clients, color coded by cluster label. """
    df = evaluation_vectors.T.melt(ignore_index=False).reset_index().set_index('client_id')
    df['label'] = -1

    for i, clus in enumerate(clusters):
        df.loc[clus, 'label'] = i

    fig = px.line(df.reset_index().sort_values(evaluation_vectors.columns.name),
                  y='value',
                  color='label',
                  x=evaluation_vectors.columns.name,
                  line_group='client_id',
                  template='plotly_white',
                  labels={'tree_num': 'Tree in global model', 'label': 'Cluster'}
                  )
    fig.show()

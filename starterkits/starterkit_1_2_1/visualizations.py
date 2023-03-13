# ©, 2022, Sirris
# owner: HCAB

import re
import pandas as pd
import matplotlib.pyplot as plt
from ipywidgets import interact
from IPython.utils import io

from starterkits.visualization import vis_plotly_widgets as vpw
from starterkits.starterkit_1_2_1.support import LstmApp

import cufflinks as cf
cf.go_offline(connected=True)
cf.set_config_file(colorscale='plotly', world_readable=True)


def interactive_plot(test_df):
    filter_columns = [x for x in test_df if re.findall(string=x, pattern='s[0-9]{1,2}')]
    plot_df = test_df.melt(id_vars=['id', 'cycle'], value_vars=filter_columns).pivot_table(index=[ 'variable', 'cycle'], columns='id', values='value' ).reset_index()
    plot_df.columns = [str(x) for x in plot_df.columns]
    plot_timeseries, var_selector, date_range_slider, filter_dropdown = vpw.timeseries_with_variable_filter(
        df=plot_df, filters=filter_columns, temporal=False, filter_column='variable', 
        description='ID', filter_description='Variable', 
        columns=test_df.id.astype(str).unique(),
        kwargs_ts={'kwargs_subplots': {'shared_yaxes': True}},
        time_index='cycle', filter_value='s11', kwargs_layout={'title':None}, add_median_line=True
    )

    interact(plot_timeseries,cols=var_selector, period=date_range_slider, filter_id = filter_dropdown);
    return date_range_slider
    
    
def start_dashboard(Xtrain, sequence_cols, test_df, mode='inline', PORT=8050, HOST='0.0.0.0'):
    if mode == 'external':
        # HOST = '127.0.0.1' 
        with io.capture_output() as captured:
            app = LstmApp(Xtrain, sequence_cols, test_df)
            ts_app = app.get_app()
            ts_app.run_server(mode='external', host=HOST, port=PORT)
        print(f'Dash app running on http://{HOST}:{PORT}')
    elif mode == 'internal': 
        with io.capture_output() as captured:
            app = LstmApp(Xtrain, sequence_cols, test_df)
            ts_app = app.get_app()
            ts_app.run_server(mode='inline')
        print(f'Dash app running on http://{HOST}:{PORT}')
    else:
        raise ValueError('mode must be external or internal')



def plot_label_frequency(label_array):
    pd.DataFrame(label_array).astype(int).astype(str)[0].value_counts(normalize=True).plot(kind='bar');
    plt.xlabel('label',fontsize=15);
    plt.ylabel('Frequency',fontsize=15);
 
 
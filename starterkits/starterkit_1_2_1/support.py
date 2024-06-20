# ©, 2022, Sirris
# owner: HCAB

import dash
from dash.dependencies import Input, Output, State
from dash import callback_context
import dash_bootstrap_components as dbc
from dash import dcc, html
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tempfile
import zipfile

from starterkits import pipeline

from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score
#from flask_caching import Cache
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras import backend as K
import tensorflow as tf
from plotly.figure_factory import create_annotated_heatmap
import base64
import json
import pickle



def get_data(DATA_PATH, force=False):
    """Extract dataset from raw data file in server and return subset dataset.

    This dataset will be saved locally.
    returns: train_df, test_df, truth_df from PM_data
    """ 

    zip_fname = pipeline.download_from_repo(
            'SK_1_2_1',
            'PM_data.zip',
            str(DATA_PATH / 'SK_1_2_1' / 'PM_data.zip'),
            force=force)

    archive = zipfile.ZipFile(zip_fname)
    for file in archive.namelist():
        archive.extract(file, str(DATA_PATH) + '/SK_1_2_1/')
    
    folder_path = os.path.join(DATA_PATH, 'SK_1_2_1')
    folder_path = os.path.join(folder_path, 'PM_data')
    train_path = os.path.join(folder_path, 'PM_train.txt')
    test_path = os.path.join(folder_path, 'PM_test.txt')
    truth_path = os.path.join(folder_path, 'PM_truth.txt')
    
    train_df = pd.read_csv(train_path, sep=' ')
    test_df = pd.read_csv(test_path, sep=' ')
    truth_df = pd.read_csv(truth_path, sep=' ')
   

    return train_df, test_df, truth_df


def load_data(DATA_PATH, force=False):
    """Reads in data and does some preprocessing. The function also creates daily and turbine-specific datasets

    dataset can be downloaded from
    # https://opendata-renewables.engie.com/explore/dataset/d543716b-368d-4c53-8fb1-55addbe8d3ad/information#
    :returns ds. dataframe containing the entire dataset
    :returns dst. dataframe with data for a single turbine
    :returns df_missing. dataframe with fake missing data for imputation section
    :returns missing_events. dataframe with summary of missing data events
    :returns missing_events_sum. dataframe with summary of missing data events per duration
    """
    def preprocess(df):
        df.drop(df.columns[[26, 27]], axis=1, inplace=True)
        df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                                 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                                 's15', 's16', 's17', 's18', 's19', 's20', 's21']
        df = df.sort_values(['id','cycle'])
        return df

    train_df, test_df, truth_df = get_data(DATA_PATH, force=force)
    
    train_df = preprocess(train_df)
    test_df = preprocess(test_df)
    truth_df.drop(truth_df.columns[[1]], axis=1, inplace=True)
        
    return train_df, test_df, truth_df



tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# Fixing the used seeds for reproducability of the experiments
np.random.seed(1234)
PYTHONHASHSEED = 0

BATCH_SIZE = 200

TEMP_DIR = tempfile.TemporaryDirectory().name
os.mkdir(TEMP_DIR)

def create_radios():
    radios_input = dbc.FormGroup(
        [
            dbc.Label("Number of intermediate layers", html_for="radioitems-number-layers", width=5),
            dbc.Col(
                dbc.RadioItems(
                    id="radioitems-number-layers",
                    options=[
                        {"label": "1 intermediate layer", "value": 1},
                        {"label": "2 intermediate layers", "value": 2},
                        {"label": "3 intermediate layers", "value": 3},
                    ],
                    value=1
                ),
                width=7,
            ),
        ],
        row=True,
    )
    return (radios_input)


def plot_cm(cm):
    df_cm = pd.DataFrame(cm, index=['no fail', 'fail'], columns=['no fail', 'fail'])
    plt.figure(figsize=(12, 8))
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=12)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=12)
    plt.ylabel('True label', fontsize=16)
    plt.xlabel('Predicted label', fontsize=16)
    plt.suptitle('Confusion matrix of the model', fontsize=18)
    plt.show()


def get_tab1_content():
    tab1_content = dbc.Card(
        dbc.CardBody([

            dbc.Card([
                dbc.CardHeader("Select parameters for model building"),
                dbc.CardBody([
                    create_radios(),
                    html.Div(id='layer_sizes'),
                    dbc.FormGroup([
                        dbc.Label('Epochs to train:', width=5),
                        dbc.Col(
                            dcc.Slider(
                                id='epoch-slider',
                                min=3,
                                max=21,
                                marks={i: '{}'.format(i) for i in range(3, 22, 3)},
                                step=1,
                                value=10), width=4
                        ),
                        dbc.Col(html.Div(id='print-epochs'), width=2)
                    ], row=True),
                    dbc.FormGroup([
                        dbc.Label('Training sequence length:', width=5),
                        dbc.Col(
                            dcc.Slider(
                                id='sequence_length',
                                min=10,
                                max=70,
                                marks={i: f'{i}' for i in range(10, 70, 5)},
                                step=1,
                                value=30), width=4
                        ),
                        dbc.Col(html.Div(id='print-sequence-length'), width=2)
                    ], row=True),
                    dbc.FormGroup([
                        dbc.Label('Dropout:', width=5),
                        dbc.Col(
                            dcc.Slider(
                                id='dropout-slider',
                                min=0.01,
                                max=0.9,
                                marks={i / 10: f'{round(i / 10, 1)}' for i in range(0, 11, 2)},
                                step=0.01,
                                value=0.2), width=4
                        ),
                        dbc.Col(html.Div(id='print-dropout'), width=2)
                    ], row=True),
                    dbc.Row(html.Div(id='model_img'), justify="center"),
                ]),

            ]),
            dbc.Card([
                dbc.CardHeader('Execute training'),
                dbc.CardBody([
                    html.Div([dbc.Row([
                        dbc.Col(dbc.Button('Train model', id='initialize_model', className="mr-2"),
                                width={"size": 5}),
                        dcc.Loading(
                            id="loading-1",
                            type="default",
                            children=dbc.Col(html.Span(id='model_initialized', ), width=4)
                        ),
                    ], justify="start", )
                    ]),
                ])
            ]),
        ]),
        className="mt-4",
        style={"width": "110%"}
    )
    return(tab1_content)


def get_tab2_content():
    tab2_content = dbc.Card(

        dbc.CardBody([
            dbc.Card([
                dbc.CardHeader('Evaluate trained model'),
                dbc.CardBody([
                    html.Div([
                        dbc.Row(dbc.Col(html.P(''))),
                        dbc.Row([
                            dbc.Col(html.Div(id='button-eval'), width=5),
                            dcc.Loading(
                                id="loading-2",
                                type="default",
                                children=html.Div(id='df-cm', hidden=True)
                            )
                        ], justify="start")]),
                ])
            ]),
            html.Div(id='acc_table', hidden=True),
            dbc.Card([
                dbc.CardHeader('Evaluation results'),
                dbc.CardBody([
                    dbc.Row([dbc.Col(html.Div(id='plot_cm'), width=10)], justify="center"),
                    dbc.Row([
                        dbc.Col(html.Div(id='show-table')),
                    ])
                ])
            ])
        ])
    )
    return(tab2_content)


def get_tab3_content():
    """
    Create test tab
    """
    tab3_content = dbc.Card(

        dbc.CardBody([

            dbc.Card([
                dbc.CardHeader("Select model for evaluation"),
                dbc.CardBody([
                    dbc.Row(dbc.Col(html.Div(id='select_test_model'), width=5)),
                    dbc.Row(dbc.Col(html.P(''))),
                    dbc.Row(dbc.Col(html.Div(id='button-select-model'), width=5)),
                ]),
            ]),
            dbc.Card(
                [
                    dbc.CardHeader("Evaluation results on the test data set"),
                    dbc.CardBody(
                        [
                            dbc.Row(
                                dbc.Col(
                                    dcc.Loading(
                                        id="loading-4",
                                        type="default",
                                        children=html.Div(id='test-results')
                                    )
                                )
                            ),
                            dbc.Row(
                                dbc.Col([
                                    html.Div(
                                        [
                                            html.Div(id='collapse_button'),
                                            dbc.Collapse(
                                                dcc.Loading(
                                                    id="loading-3",
                                                    type="default",
                                                    children=html.Div(id='all_models')

                                                ), id="collapse",
                                            ),
                                        ]
                                    )
                                ])
                            )
                        ])
                ]
            )
        ]),
        className="mt-4",
        style={"width": "110%"}
    )
    return(tab3_content)


def gen_sequence(id_df, seq_length, seq_cols):
    """
    Function to reshape the data in into a 3-dimensional (samples, time steps, features) matrix
    """
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        yield data_array[start:stop, :]


def gen_labels(id_df, seq_length, label):
    data_array = id_df[label].values
    num_elements = data_array.shape[0]
    return data_array[seq_length:num_elements, :]


class LstmApp():

    def __init__(self, Xtrain, sequence_cols, test_df):
        """
        Parameters:
        Xtrain: array, training data
        sequence_cols: list of str, defining which columns are features
        test_df: array, test data
        """
        self.Xtrain = Xtrain
        self.sequence_cols = sequence_cols
        self.test_df = test_df
        self.clean_up()

    def clean_up(self):
        try:
            os.remove(TEMP_DIR, 'model.png')
        except:
            pass
        try:
            os.remove(TEMP_DIR, 'scores.pkl')
        except:
            pass

    def create_df(self, df, w1):
        df['RUL_label'] = np.where(df['RUL'] <= w1, 1, 0)
        # MinMax normalization
        df['cycle_norm'] = df['cycle']
        cols_normalize = df.columns.difference(['id', 'cycle', 'RUL'])
        min_max_scaler = preprocessing.MinMaxScaler()
        norm_Xtrain = pd.DataFrame(min_max_scaler.fit_transform(df[cols_normalize]),
                                   columns=cols_normalize,
                                   index=df.index)
        join_df = df[df.columns.difference(cols_normalize)].join(norm_Xtrain)
        df = join_df.reindex(columns=df.columns)

        seq_gen = (list(gen_sequence(df[df['id'] == id_], w1, self.sequence_cols))
                   for id_ in df['id'].unique())

        # Convert the generated sequences to a NumPy array
        seq_array = np.concatenate(list(seq_gen)).astype(np.float32)
        #     printmd(f'''This results in a 3-dimensional input matrix with {seq_array.shape[0]} samples,
        #     {seq_array.shape[1]} time steps and {seq_array.shape[2]} features (including 21 sensor values,
        #     the 3 settings parameters and the normalised current cycle).''')
        label_gen = [gen_labels(df[df['id'] == id_], w1, ['RUL_label'])
                     for id_ in df['id'].unique()]
        label_array = np.concatenate(label_gen).astype(np.float32)
        return (seq_array, label_array)


    def get_app(self):

        # ts_app = JupyterDash(external_stylesheets=[dbc.themes.BOOTSTRAP],
        #                      suppress_callback_exceptions=True)
        ts_app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP], 
                           suppress_callback_exceptions=True)
        ts_app.layout = dbc.Container([
            dbc.Tabs([
                dbc.Tab(get_tab1_content(), label='Training'),
                dbc.Tab(get_tab2_content(), label='Evaluation'),
                dbc.Tab(get_tab3_content(), label='Testing')
            ])

        ])

        #CACHE_CONFIG = {
        #    # try 'filesystem' if you don't want to setup redis
        #    'CACHE_TYPE': 'filesystem',
        #    'CACHE_DIR': './outputs',
        #    'CACHE_THRESHOLD': 200
        #    # 'CACHE_REDIS_URL': os.environ.get('REDIS_URL', 'redis://localhost:6379')
        #}
        #cache = Cache()
        #cache.init_app(ts_app.server, config=CACHE_CONFIG)

        @ts_app.callback(
            Output('print-epochs', 'children'),
            Input('epoch-slider', 'value')
        )
        def print_epochs(n):
            return (html.Div(n))

        @ts_app.callback(
            Output('print-dropout', 'children'),
            Input('dropout-slider', 'value')
        )
        def print_epochs(n):
            return (html.Div(n))

        @ts_app.callback(
            Output('print-sequence-length', 'children'),
            Input('sequence_length', 'value')
        )
        def print_epochs(n):
            return (html.Div(n))

        @ts_app.callback(
            Output('layer_sizes', 'children'),
            [Input('radioitems-number-layers', 'value')])
        def construct_layer_size(num_layers):
            form = dbc.Row(
                [
                    dbc.Col(
                        dbc.FormGroup(
                            [
                                dbc.Label(f"Intermediate Layer {i + 1}", html_for=f"layer_label_{i}"),
                                dbc.Input(
                                    type="number",
                                    id=f"layer_size_{i}",
                                    placeholder=f"e.g. {int(400 / 2 / (i + 1))}",
                                    debounce=True,
                                    disabled=i >= num_layers,
                                    min=5, max=400
                                ),
                            ]
                        ),
                        width=int(4),
                    ) for i in range(3)
                ],
                form=True,
            )
            return (form)

        #@cache.memoize()
        def global_model_store(size_1, size_2, size_3, num_layers, seq_len, n, epochs, dropout):
            # simulate expensive query
            sizes = [size_1, size_2, size_3, None]
            changed_ids = [p['prop_id'].split('.')[0] for p in callback_context.triggered]
            button_pressed = 'initialize_model' in changed_ids

            seq_array, label_array = self.create_df(self.Xtrain, seq_len)
            if n is None:
                return (True)
            if button_pressed:
                # build the network

                nb_features = seq_array.shape[2]
                nb_out = label_array.shape[1]
                K.clear_session()
                # with sess.as_default():
                model_ = Sequential(name='Sequential_Layer')

                for i in range(num_layers):
                    if sizes[i]:
                        if i == 0:
                            model_.add(LSTM(
                                input_shape=(seq_len, nb_features),
                                units=sizes[i],
                                return_sequences=bool(sizes[i + 1] is not None)))
                        else:
                            model_.add(LSTM(
                                units=sizes[i],
                                return_sequences=bool(sizes[i + 1] is not None)))
                        model_.add(Dropout(dropout))

                model_.add(Dense(units=nb_out, activation='sigmoid'))

                model_.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                # #print(model.summary())
                # #print("Train model...")

                model_.fit(seq_array, label_array, epochs=epochs, batch_size=BATCH_SIZE,
                           validation_split=0.05,
                           verbose=0,
                           callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                    min_delta=0,
                                                                    patience=0,
                                                                    verbose=0, mode='auto')])

                model_.save(
                    os.path.join(
                        TEMP_DIR,
                        f'model_{size_1}_{size_2}_{size_3}_{num_layers}_{epochs}_{dropout}_{seq_len}.h5'))
                return (True)
            elif 'eval-button' in changed_ids:
                return (True)
            else:
                return (False)

        def model(size_1=None, size_2=None, size_3=None, num_layers=None, seq_len=None, n=None,
                  epochs=None, dropout=None):
            return (global_model_store(size_1, size_2, size_3, num_layers, seq_len, n, epochs, dropout))

        @ts_app.callback(
            Output('model_img', 'children'),
            [Input(f"layer_size_{i}", "value") for i in range(3)],
            Input('radioitems-number-layers', 'value'),
            Input('dropout-slider', 'value'),
            Input('sequence_length', 'value'),
        )
        def create_img_model(size_1, size_2, size_3, num_layers, dropout, seq_len):
            sizes = [size_1, size_2, size_3, None]
            # p = Path('.')
            seq_array, label_array = self.create_df(self.Xtrain, seq_len)
            # local_path = p.absolute()
            # build the network
            nb_features = seq_array.shape[2]
            nb_out = label_array.shape[1]
            K.clear_session()
            # with sess.as_default():

            model_ = Sequential(name='Sequential_Layer')

            # Put in seperate function and cell
            for i in range(num_layers):
                if sizes[i]:
                    if i == 0:
                        model_.add(LSTM(
                            input_shape=(seq_len, nb_features),
                            units=sizes[i], name=f'LSTM_{i}',
                            return_sequences=bool(sizes[i + 1] is not None)
                        ))
                    else:
                        model_.add(LSTM(
                            units=sizes[i], name=f'LSTM_{i}',
                            return_sequences=bool(sizes[i + 1] is not None)))
                    model_.add(Dropout(dropout, name=f'Dropout_{i}'))

            if sizes[0] is not None:
                model_.add(Dense(units=nb_out, activation='sigmoid'))
                model_.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

                image_filename = os.path.join(TEMP_DIR, 'model.png')
                # pm(model, show_shapes=True, show_layer_names=True, to_file=image_filename)
                tf.keras.utils.plot_model(model_, show_shapes=True, show_layer_names=True, to_file=image_filename)

                encoded_image = base64.b64encode(open(image_filename, 'rb').read())
                return (html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()),
                                 style={
                                     "frameborder": "1px",
                                     "text-align": "center"
                                 }))
            else:
                return (None)

        @ts_app.callback(
            Output('model_initialized', 'children'),
            Input('initialize_model', 'n_clicks'),
            [Input(f"layer_size_{i}", "value") for i in range(3)], Input('radioitems-number-layers', 'value'),
            Input('epoch-slider', 'value'), Input('dropout-slider', 'value'), Input('sequence_length', 'value')
        )
        def build_model(n, size_1, size_2, size_3, num_layers, epochs, dropout, seq_len):
            changed_ids = [p['prop_id'].split('.')[0] for p in callback_context.triggered]
            button_pressed = 'initialize_model' in changed_ids

            bb = None
            model_trained = None
            if button_pressed:
                model_trained = global_model_store(size_1, size_2, size_3, num_layers, seq_len, n, epochs, dropout)
            if model_trained:
                bb = [dbc.Badge("Success - Model trained", color="success")]
            return (bb)

        @ts_app.callback(
            Output('button-eval', 'children'),
            Input('model_initialized', 'children')
        )
        def eval_button(but):
            changed_ids = [p['prop_id'].split('.')[0] for p in callback_context.triggered]
            button_pressed = 'model_initialized' in changed_ids
            # bb = None
            if but and button_pressed:
                bb = dbc.Button('Evaluate model', id='eval-button', className="mr-1", n_clicks=None)
            else:
                bb = dbc.Button('Evaluate model', id='eval-button', className="mr-1", disabled=True)
            return (bb)

        @ts_app.callback(
            Output('button-select-model', 'children'),
            Input('selected_test', 'value')
        )
        def test_model_button(but):
            # bb = None
            if but is not None:
                bb = dbc.Button('Evaluate model', id='select-button', className="mr-1", n_clicks=None)
            else:
                bb = dbc.Button('Evaluate model', id='select-button', className="mr-1", disabled=True)
            return (bb)

        @ts_app.callback(
            Output('collapse_button', 'children'),
            # Input('collapse_button', 'value'),
            Input('test-results', 'children')
        )
        def coll_button_(but):
            # bb = None
            if but is not None:
                bb = dbc.Button(
                    "Show metrics from all trained models",
                    id="collapse-button",
                    className="mb-3",
                )
            else:
                bb = dbc.Button(
                    "Show metrics from all trained models",
                    id="collapse-button",
                    className="mb-3",
                    disabled=True
                )
            return (bb)

        @ts_app.callback(
            Output('df-cm', 'children'),
            Input('eval-button', 'n_clicks'),
            [State(f"layer_size_{i}", "value") for i in range(3)],
            State('radioitems-number-layers', 'value'),
            State('epoch-slider', 'value'), State('dropout-slider', 'value'),
            State('df-cm', 'children'), State('sequence_length', 'value')
        )
        def eval_model(n, size_1, size_2, size_3, num_lays, epochs, dropout, eval_data, seq_len):
            changed_ids = [p['prop_id'].split('.')[0] for p in callback_context.triggered]
            button_pressed = 'eval-button' in changed_ids
            seq_array, label_array = self.create_df(self.Xtrain, seq_len)
            if (button_pressed):
                if n is not None:
                    model_trained = model(size_1, size_2, size_3, num_lays, seq_len, n, epochs, dropout)
                    if model_trained:
                        K.clear_session()
                        try:
                            mod = tf.keras.models.load_model(
                                os.path.join(
                                    TEMP_DIR,
                                    f'model_{size_1}_{size_2}_{size_3}_{num_lays}_{epochs}_{dropout}_{seq_len}.h5'))
                        except Exception:
                            raise ValueError('Something went wrong')
                        scores = mod.evaluate(seq_array, label_array, verbose=0, batch_size=BATCH_SIZE)
                        y_pred = (mod.predict(seq_array, verbose=0, batch_size=BATCH_SIZE) > 0.5).astype("int32")
                        y_true = label_array
                        cm = confusion_matrix(y_true, y_pred)
                        precision = precision_score(y_true, y_pred)
                        recall = recall_score(y_true, y_pred)
                        df_cm = pd.DataFrame(cm, index=['no fail', 'fail'], columns=['no fail', 'fail'])

                        return (json.JSONEncoder().encode(dict(cm=df_cm.to_dict(),
                                                               scores={'accuracy': float(scores[1]),
                                                                       'precision': precision,
                                                                       'recall': recall
                                                                       })))
                elif eval_data is not None:
                    return (eval_data)
            elif eval_data is not None:
                return (eval_data)
            else:
                return (None)

        @ts_app.callback(Output('plot_cm', 'children'), Input('df-cm', 'children'))
        def plot_model_(eval_data):
            if eval_data is not None:
                eval_data = json.JSONDecoder().decode(eval_data)
                df_cm = pd.DataFrame.from_dict(eval_data['cm'])
                fig = create_annotated_heatmap(z=df_cm.values, x=df_cm.index.tolist(),
                                               y=df_cm.columns.tolist(), showscale=False,
                                               colorscale='Blues')

                fig.update_xaxes(title='Predicted label')
                fig.update_yaxes(title='True label')

                bb = dcc.Graph(figure=fig, id='plotted-cm')
                return (bb)
            else:
                return (None)

        @ts_app.callback(Output("acc_table", "children"),
                         Input("df-cm", "children"),
                         [State(f"layer_size_{i}", "value") for i in range(3)],
                         State('radioitems-number-layers', 'value'),
                         State('epoch-slider', 'value'),
                         State('dropout-slider', 'value'),
                         State('sequence_length', 'value'),
                         State("acc_table", "children"))  # + [State(column, "value") for column in columns])
        def append_(eval_data, size1, size2, size3, num_layers, epochs, dropout, seq_len, data_2):
            if data_2 is not None:

                if eval_data is not None:
                    data = json.JSONDecoder().decode(data_2)
                    eval_data = json.JSONDecoder().decode(eval_data)
                    temp_df = pd.DataFrame(
                        data={'tp': [eval_data['cm']['no fail']['no fail']],
                              'fp': [eval_data['cm']['no fail']['fail']],
                              'tn': [eval_data['cm']['fail']['fail']],
                              'fn': [eval_data['cm']['fail']['no fail']]})
                    temp_data = pd.DataFrame(data=data)[['tp', 'fp', 'tn', 'fn']]
                    temp_data = temp_data.append(temp_df)
                    if not temp_data.duplicated().iloc[-1]:
                        data.append({'# layers': num_layers,
                                     'layer size 1': size1,
                                     'layer size 2': size2 if num_layers > 1 else '-',
                                     'layer size 3': size3 if num_layers > 2 else '-',
                                     'sequence length': seq_len,
                                     'dropout': dropout,
                                     '#epochs': epochs,
                                     'accurracy': round(eval_data['scores']['accuracy'], 5),
                                     'precision': round(eval_data['scores']['precision'], 3),
                                     'recall': round(eval_data['scores']['recall'], 3),
                                     'tp': eval_data['cm']['no fail']['no fail'],
                                     'fp': eval_data['cm']['no fail']['fail'],
                                     'tn': eval_data['cm']['fail']['fail'],
                                     'fn': eval_data['cm']['fail']['no fail']
                                     })
                        data = json.JSONEncoder().encode(data)

                    else:
                        data = json.JSONDecoder().decode(data_2)
                        data = json.JSONEncoder().encode(data)
                else:
                    data = json.JSONDecoder().decode(data_2)
                    data = json.JSONEncoder().encode(data)
            elif eval_data is not None:
                eval_data = json.JSONDecoder().decode(eval_data)
                data = [{'# layers': num_layers,
                         'layer size 1': size1,
                         'layer size 2': size2 if num_layers > 1 else '-',
                         'layer size 3': size3 if num_layers > 2 else '-',
                         'sequence length': seq_len,
                         'dropout': dropout,
                         '#epochs': epochs,
                         'accurracy': round(eval_data['scores']['accuracy'], 5),
                         'precision': round(eval_data['scores']['precision'], 3),
                         'recall': round(eval_data['scores']['recall'], 3),
                         'tp': eval_data['cm']['no fail']['no fail'],
                         'fp': eval_data['cm']['no fail']['fail'],
                         'tn': eval_data['cm']['fail']['fail'],
                         'fn': eval_data['cm']['fail']['no fail']
                         }]
                data = json.JSONEncoder().encode(data)
            else:
                return (None)
            return (data)

        @ts_app.callback(Output('show-table', 'children'), Input("acc_table", "children"))
        def show_table(data):
            if data is not None:
                import pdb;pdb.set_trace()
                cols = ['model id', '# layers', 'layer size 1', 'layer size 2',
                        'layer size 3', 'sequence length', 'dropout', '#epochs', 'accurracy', 'precision', 'recall']
                data = pd.DataFrame.from_dict(json.JSONDecoder().decode(data))
                data = data.drop_duplicates()
                data['model id'] = list(range(data.shape[0]))
                data = data[cols]
                return (dbc.Table.from_dataframe(data, striped=True, bordered=True, hover=True))
            else:
                return (None)

        @ts_app.callback(Output('select_test_model', 'children'),
                         Input('acc_table', 'children'))
        def select_test_m(data):
            if data is not None:
                data = pd.DataFrame.from_dict(json.JSONDecoder().decode(data))
                data = data.drop_duplicates()
                data['model id'] = list(range(data.shape[0]))

                options = [{'label': f'Model id {i}', 'value': i} for i in data['model id']]
                select = dbc.Select(
                    id="selected_test",
                    options=options,
                    value=0
                )
                return (select)
            else:
                ret = dbc.Alert(
                    "Please train a model first...",
                    id="alert-fade",
                    dismissable=True,
                    is_open=True,
                    fade=True,
                ),
                return (ret)

        @ts_app.callback(Output('test-results', 'children'),
                         Input('select-button', 'n_clicks'),
                         State('selected_test', 'value'),
                         State("acc_table", "children"))
        def test_model(clicked, model_id, data):
            if data is not None and clicked is not None:

                data = pd.DataFrame.from_dict(json.JSONDecoder().decode(data))
                data = data.drop_duplicates()
                data['model id'] = list(range(data.shape[0]))
                data['model id'] = data['model id'].astype(type(model_id))
                data = data[data['model id'].isin([model_id])]
                sequence_length = data['sequence length'].values[0]
                # model_trained = model(size_1, size_2, size_3, num_layers, n, epochs, dropout)

                K.clear_session()
                try:
                    size1 = data['layer size 1'].values[0] if data['layer size 1'].values[0] != '-' else 'None'
                    size2 = data['layer size 2'].values[0] if data['layer size 2'].values[0] != '-' else 'None'
                    size3 = data['layer size 3'].values[0] if data['layer size 3'].values[0] != '-' else 'None'
                    l1 = data['# layers'].values[0]
                    epoch = data['#epochs'].values[0]
                    dropout = data['dropout'].values[0]
                    sl = data['sequence length'].values[0]
                    mod = tf.keras.models.load_model(
                        os.path.join(
                            TEMP_DIR,
                            f'model_{size1}_{size2}_{size3}_{l1}_{epoch}_{dropout}_{sl}.h5'))
                except Exception:
                    raise ValueError('Something went wrong')

                seq_array_test_last = ([self.test_df[self.test_df['id'] == id_]
                                        [self.sequence_cols].values[-sequence_length:]
                                        for id_ in self.test_df['id'].unique() if
                                        len(self.test_df[self.test_df['id'] == id_]) >= sequence_length])

                seq_array_test_last = np.asarray(seq_array_test_last).astype(np.float32)

                y_mask = ([len(self.test_df[self.test_df['id'] == id_])
                           >= sequence_length for id_ in self.test_df['id'].unique()])
                label_array_test_last = self.test_df.groupby('id')['RUL_label'].nth(-1)[y_mask].values
                label_array_test_last = label_array_test_last.reshape(label_array_test_last.shape[0], 1).astype(
                    np.float32)
                # Compute accuracy
                scores_test = mod.evaluate(seq_array_test_last, label_array_test_last, verbose=0)

                card = html.P(f"For model with model id {model_id}, the accuracy is {round(scores_test[1] * 100, 2)}%"),
                return (card)
            else:
                return (None)

        @ts_app.callback(
            Output("collapse", "is_open"),
            [Input("collapse-button", "n_clicks")],
            [State("collapse", "is_open")],
        )
        def toggle_collapse(n, is_open):
            if n:
                return not is_open
            return is_open

        @ts_app.callback(
            Output("all_models", "children"),
            [Input("collapse-button", "n_clicks")],
            [State("collapse", "is_open"),
             State("acc_table", "children")
             ],
        )
        def toggle_collapse_(n, is_open, data):

            if not is_open:
                if data is not None:
                    data = pd.DataFrame.from_dict(json.JSONDecoder().decode(data))
                    data = data.drop_duplicates()
                    data['model id'] = list(range(data.shape[0]))

                    eval_df_test = pd.DataFrame()
                    for model_id in data['model id'].unique():
                        data_selected = data[data['model id'].isin([model_id])]
                        K.clear_session()
                        seq_len = data_selected['sequence length'].values[0]
                        try:
                            size1 = data_selected['layer size 1'].values[0] if data_selected['layer size 1'].values[
                                                                                   0] != '-' else 'None'
                            size2 = data_selected['layer size 2'].values[0] if data_selected['layer size 2'].values[
                                                                                   0] != '-' else 'None'
                            size3 = data_selected['layer size 3'].values[0] if data_selected['layer size 3'].values[
                                                                                   0] != '-' else 'None'

                            l1 = data_selected['# layers'].values[0]
                            epochs = data_selected['#epochs'].values[0]
                            do = data_selected['dropout'].values[0]
                            sl = data_selected['sequence length'].values[0]
                            mod = tf.keras.models.load_model(
                                os.path.join(
                                    TEMP_DIR,
                                    f'model_{size1}_{size2}_{size3}_{l1}_{epochs}_{do}_{sl}.h5'))
                        except Exception:
                            raise ValueError('Something went wrong')

                        seq_array_test_last = [
                            self.test_df[self.test_df['id'] == id_][self.sequence_cols].values[-seq_len:]
                            for id_ in self.test_df['id'].unique() if
                            len(self.test_df[self.test_df['id'] == id_]) >= seq_len]

                        seq_array_test_last = np.asarray(seq_array_test_last).astype(np.float32)

                        y_mask = [len(self.test_df[self.test_df['id'] == id_]) >= seq_len for id_ in
                                  self.test_df['id'].unique()]
                        label_array_test_last = self.test_df.groupby('id')['RUL_label'].nth(-1)[y_mask].values
                        label_array_test_last = (label_array_test_last
                            .reshape(label_array_test_last.shape[0], 1)
                            .astype(np.float32))
                        # Compute accuracy
                        scores_test = mod.evaluate(seq_array_test_last, label_array_test_last, verbose=0)
                        y_pred = (mod.predict(seq_array_test_last, verbose=0, batch_size=BATCH_SIZE) > 0.5).astype("int32")
                        y_true = label_array_test_last
                        # Compute precision and recall
                        precision = round(precision_score(y_true, y_pred), 3)
                        recall = recall_score(y_true, y_pred)
                        f1 = f1_score(y_true, y_pred)
                        eval_df_test = eval_df_test.append(
                            pd.DataFrame(data={'model id': [model_id], 'Accuracy': [round(scores_test[1] * 100, 3)],
                                               'Precision': [precision], 'Recall': recall, 'F1-score': [f1]}))
                        pickle.dump(eval_df_test, open(os.path.join(TEMP_DIR, 'scores.pkl'), 'wb'))
                    return (dbc.Table.from_dataframe(eval_df_test, striped=True, bordered=True, hover=True))
            else:
                return (None)

        return (ts_app)


def get_test_scores():
    try:
        with open(os.path.join(TEMP_DIR, 'scores.pkl'), 'rb') as f:
            scores = pickle.load(f)
            scores['model_type'] = 'LSTM'
    except:
        scores = pd.DataFrame()

    best_ = pd.DataFrame([['-', 94.0, 0.952381, 0.8, 0.869565, 'Two-Class Neural Network (best performing model from [1])']],
                         columns = ['model id','Accuracy', 'Precision', 'Recall', 'F1-score', 'model_type'])
    scores = pd.concat([scores, best_])
    scores = scores.set_index('model_type')
    return(scores)

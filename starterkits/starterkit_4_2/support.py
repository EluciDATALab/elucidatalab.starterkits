import random
import gc
import zipfile
import os
import pandas as pd
import numpy as np
import seaborn as sb
import plotly.express as px
import matplotlib.pyplot as plt
import pyswarms.backend as ps
from pyswarms.backend.topology import Star

from tqdm.notebook import tqdm
from starterkits import pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, pairwise_distances, davies_bouldin_score, silhouette_score
from scipy.spatial.distance import cdist
from IPython.display import display, clear_output

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def get_data(DATA_PATH, force=False):
    """Exctract dataset from raw data file in server and return subset dataset.

    This dataset will be saved locally.
    returns: subset UK household dataset
    """
    def extract_csv_file(fname):
        zip_fname = pipeline.download_from_repo(
            'SK_4_2',
            'household_energy_consumption.zip',
            str(DATA_PATH / 'SK_4_2' / 'household_energy_consumption.zip'),
            force=force)
        with zipfile.ZipFile(zip_fname, 'r') as zip_:
            zip_.extract('household_energy_consumption.csv', path=str(DATA_PATH / 'SK_4_2'))
        os.remove(zip_fname)
        return fname
    fname = (pipeline.File(extract_csv_file,
                           str(DATA_PATH / 'SK_4_2' / 'household_energy_consumption.csv'))
             .make(force=force))

    df = pd.read_csv(fname, index_col='Unnamed: 0')

    return df

#
# def series_to_supervised(series=None, time_lags=None, output_length=None):
#     """ Transform a time series dataset into a supervised learning dataset.   """
#     cols = []
#     # input sequence (t-n, ... t-1)
#     for i in time_lags:
#         cols.append(series.shift(i))
#     # forecast sequence (t, t+1, ... t+n)
#     for i in range(0, output_length):
#         cols.append(series.shift(-i))
#     # put it all together
#     agg = pd.concat(cols, axis=1)
#     agg.index = series.index
#     agg.dropna(inplace=True)
#
#     lag_names = [str(30 * -i) for i in range(1, 2 + 1 + 10)] + \
#                 ['previous_day-30', 'previous_day', 'previous_day+30'] + \
#                 ['previous_week-30', 'previous_week', 'previous_week+30'] + \
#                 ['label']
#     agg.columns = lag_names
#
#     return agg


def series_to_supervised(series, time_lags):
    """Transform a time series dataset into a supervised learning dataset.

    Args:
        series (pd.Series): Time series data.
        time_lags (list of int): List of time lags for the input sequence.

    Returns:
        pd.DataFrame: Supervised learning dataset.
    """
    # Generating input features using time lags
    cols = [series.shift(i) for i in time_lags]
    # Adding the label column (forecasted value)
    cols.append(series.shift(-1))  # This is the target column

    # Concatenate all columns together
    agg = pd.concat(cols, axis=1)
    agg.index = series.index
    agg.dropna(inplace=True)

    # Generating column names for input features and label
    lag_names = [f"t-{i}" for i in time_lags] + ['label']
    agg.columns = lag_names

    return agg


class PreprocessConsumption:

    def __init__(self, data=None, normalize=False,
                 train_start='2012-01-01', train_end='2012-04-01',
                 test_start='2012-04-01', test_end='2012-05-01',
                 val_start='2012-05-01', val_end='2014-01-01',
                 timestamps_per_hour=2):
        self.data = data
        self.consumer_list = self.data.consumer.unique()
        self.normalize = normalize
        self.train_start = train_start
        self.train_end = train_end
        self.test_start = test_start
        self.test_end = test_end
        self.val_start = val_start
        self.val_end = val_end

        self.timestamps_per_hour = timestamps_per_hour
        self.minutes_per_timestamp = int(60 / timestamps_per_hour)
        self.previous_week_base = timestamps_per_hour * 24 * 7
        self.previous_day_base = timestamps_per_hour * 24

        self.scaler_dict = dict.fromkeys(self.consumer_list)
        self.data.timestamp = pd.to_datetime(data.timestamp)
        self.data_dict = dict.fromkeys(self.consumer_list)
        self.features_dict = dict.fromkeys(self.consumer_list)

    def add_time_features(self):
        """ Addd dow, moy and hod features"""
        self.data['dow'] = self.data['timestamp'].dt.dayofweek
        self.data['moy_cos'] = np.cos(2 * np.pi * self.data['timestamp'].dt.month / 12)
        self.data['hod_cos'] = np.cos(2 * np.pi * self.data['timestamp'].dt.hour / 23)

        self.data['moy_sin'] = np.sin(2 * np.pi * self.data['timestamp'].dt.month / 12)
        self.data['hod_sin'] = np.sin(2 * np.pi * self.data['timestamp'].dt.hour / 23)

    def convert_to_dict(self):
        """ Separate data per consumer in dictionary."""
        self.data_dict = {cons: group.set_index('timestamp')
                          for cons, group in self.data.groupby('consumer')}

    def normalize_consumption(self):
        """ Normalize raw consumption per consumer."""
        for (cons, subset) in self.data_dict.items():
            scaler = MinMaxScaler()
            subset['consumption_scaled'] = scaler.fit_transform(np.array(subset['consumption']).reshape(-1, 1))
            self.scaler_dict[cons] = scaler

    def train_test_split(self, df):
        train_set = df[(df.index >= self.train_start) & (df.index < self.train_end)]
        test_set = df[(df.index >= self.test_start) & (df.index < self.test_end)]
        val_set = df[(df.index >= self.val_start) & (df.index < self.val_end)]

        return train_set, test_set, val_set

    def create_feature_df(self):
        """ Convert series to feature set per consumer."""
        # define time lags to be features
        daily_lags = self.timestamps_per_hour * 24
        weekly_lags = daily_lags * 7
        mid_intervals = [int(-self.timestamps_per_hour / 2), 0, int(self.timestamps_per_hour / 2)]
        time_lags = list(range(1, self.timestamps_per_hour + 11)) + \
            [daily_lags + offset for offset in mid_intervals] + \
            [weekly_lags + offset for offset in mid_intervals]

        for cons, subset in self.data_dict.items():
            df = series_to_supervised(series=subset['consumption_scaled' if self.normalize else 'consumption'],
                                      time_lags=time_lags)

            # Join with hod, dow, moy without separate step
            df = df.join(subset[['hod_cos', 'hod_sin', 'moy_cos', 'moy_sin', 'dow']].loc[df.index])

            train_set, test_set, val_set = self.train_test_split(df)
            train_labels = train_set.pop('label')
            test_labels = test_set.pop('label')
            val_labels = val_set.pop('label')

            self.features_dict[cons] = {
                'train': [train_set, train_labels],
                'test': [test_set, test_labels],
                'val': [val_set, val_labels]
            }

    def extract_x_y(self, train_or_test):
        """ Return features and labels in list."""
        x = []
        y = []
        for cons in self.consumer_list:
            x.append(self.features_dict[cons][train_or_test][0])
            y.append(self.features_dict[cons][train_or_test][1])

        return x, y

    def preprocess(self):
        self.add_time_features()
        self.convert_to_dict()
        if self.normalize:
            self.normalize_consumption()
        self.create_feature_df()

        x_train, y_train = self.extract_x_y('train')
        x_test, y_test = self.extract_x_y('test')

        return x_train, y_train, x_test, y_test


class Client:
    """
    Client class
    """

    def __init__(self, client_id=None):
        """
        Initialise client class
        Parameters
        ----------
        client_id: int or str, identify client
        """

        self.tree = None
        self.train_data = None
        self.test_data = None
        self.test_labels = None
        self.train_labels = None
        self.predictions = None
        self.client_id = client_id

    def assign_data(self, x_train, y_train, x_test, y_test):
        """ Assign train and validation data."""
        # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle=False)
        self.train_data = x_train.reset_index(drop=True)
        self.train_labels = y_train.reset_index(drop=True)
        self.train_labels.columns = ['label']

        self.test_data = x_test.reset_index(drop=True)
        self.test_labels = y_test.reset_index(drop=True)
        self.test_labels.columns = ['label']

    def grow_forest(self, n_trees, tree_depth, min_samples_per_split):
        """ Grow 1 forest per fixed order."""
        forest = RandomForestRegressor(n_estimators=n_trees,
                                       max_depth=tree_depth,
                                       criterion='squared_error',
                                       min_samples_split=min_samples_per_split,
                                       bootstrap=True,
                                       max_samples=0.8,
                                       verbose=0,
                                       n_jobs=-1
                                       # random_state=1
                                       )
        forest.fit(self.train_data, self.train_labels)

        for tr in forest.estimators_:
            tr.owner = self.client_id

        forest.owners = self.client_id

        return forest.estimators_, forest

    def evaluate_trees(self, global_forest):
        """ Evaluate losses for each tree in global forest."""
        trees_and_loss = []

        for i, tree in enumerate(global_forest):
            predictions = tree.predict(self.test_data)
            mse_loss = mean_squared_error(self.test_labels, predictions)
            trees_and_loss.append((tree, mse_loss, self.client_id, tree.owner, i))

        return pd.DataFrame(trees_and_loss, columns=['tree', 'mse_loss', 'client_id', 'owners', 'tree_num'])

    def evaluate_forests(self, forest_list):
        """ Evaluate losses for each local forest. """
        forests_and_loss = []

        for i, forest in enumerate(forest_list):
            predictions = forest.predict(self.test_data)
            mse_loss = mean_squared_error(self.test_labels, predictions)
            # asymm_loss = asymmetric_cost(self.test_labels, predictions, 1)
            forests_and_loss.append((forest, mse_loss, self.client_id, forest.owners, i))

        return pd.DataFrame(forests_and_loss, columns=['forest', 'mse_loss', 'client_id', 'owners', 'forest_num'])


class FederatedForest:
    """
    Federated forest training class
    """

    def __init__(self,
                 min_samples_per_split=10,
                 num_trees_per_client=100,
                 client_ids=None,
                 tree_depth=10):
        """

        Parameters
        ----------
        client_ids: list, ids of all clients to federated across
        """
        self.clients = [Client(client_id=id_) for id_ in client_ids]
        self.min_samples_per_split = min_samples_per_split
        self.tree_depth = tree_depth
        self.num_trees_per_client = num_trees_per_client
        self.client_ids = client_ids
        self.forest = None
        self.scores = None
        self.all_forests = None
        self.trees_and_losses = None
        self.forests_and_losses = None
        self.features = None
        self.global_forest = None

    def assign_client_data(self, x_train, y_train, x_test, y_test):
        """ Assign data to client. """
        self.features = list(x_train[0].columns)

        for i, client in enumerate(self.clients):
            client.assign_data(x_train[i], y_train[i], x_test[i], y_test[i])

    def train_local_forests(self):
        """ Select fixed orders, build and evaluate trees per client and order. """
        self.all_forests = []
        for nth_client, client in tqdm(enumerate(self.clients),
                                       desc='Retrieving trees', unit='client', total=len(self.clients)):
            trees, forest = client.grow_forest(self.num_trees_per_client, self.tree_depth,
                                               self.min_samples_per_split)
            self.all_forests.append(forest)

    def construct_global_forest(self):
        """ Each local forest donates trees (size forest / # workers) to construct 1 global model. """
        # random.seed(42)
        glob_forest = np.array([
            random.sample(forest.estimators_, int(round(self.num_trees_per_client / len(self.clients), 0) + 1))
            for forest in self.all_forests])
        glob_forest = glob_forest.flatten()
        glob_forest = glob_forest[random.sample(range(glob_forest.shape[0]), self.num_trees_per_client)]
        self.global_forest = glob_forest

    def train(self):
        """ Retrieve local forests and create global forest. """
        self.train_local_forests()
        self.construct_global_forest()

    def evaluate_local_forests(self):
        """ Evaluate each local forest by each worker to create evaluation vector. """
        # self.forests_and_losses = pd.DataFrame()
        # for client in tqdm(self.clients):
        #     forests_and_loss_client = client.evaluate_forests(self.all_forests)
        #     self.forests_and_losses = self.forests_and_losses.append(forests_and_loss_client)

        evaluations = [client.evaluate_forests(self.all_forests) for client in tqdm(self.clients)]
        self.forests_and_losses = pd.concat(evaluations, ignore_index=True)

    def evaluate_global_forest(self):
        """ Evaluate each tree in global forest to create evaluation vector. """
        # self.trees_and_losses = pd.DataFrame()
        # for client in tqdm(self.clients):
        #     evaluation_vector = client.evaluate_trees(self.global_forest)
        #     self.trees_and_losses = self.trees_and_losses.append(evaluation_vector)

        evaluations = [client.evaluate_trees(self.global_forest) for client in tqdm(self.clients,
                                                                                    desc='Evaluating trees',
                                                                                    unit='client')]
        self.trees_and_losses = pd.concat(evaluations, ignore_index=True)

    def show_rmse_score(self, annotate=True):
        """ Bar plot of MSE scores."""
        df = self.scores.melt(ignore_index=False, var_name='Forest type', value_name='RMSE')
        plt.figure(figsize=(25, 5))
        ax = sb.barplot(data=df, x=df.index, y='RMSE', hue='Forest type')
        ax.set(title='RMSE scores per client for local, cluster and global forests.')
        if annotate:
            for i in ax.containers:
                ax.bar_label(i, fmt="%.4f", padding=2, fontsize=8)

    # def show_losses(self, per_client=False, loss_type=None):
    #     """ Show losses of trees."""
    #     if per_client:
    #         g = sb.catplot(data=self.all_trees,
    #                        x='order_str',
    #                        y=loss_type,
    #                        hue='order_str',
    #                        height=4,
    #                        aspect=1.5,
    #                        sharey=False,
    #                        col='client_id',
    #                        col_wrap=3,
    #                        )
    #         g.set(xticks=[], xlabel='Fixed order')
    #         g.add_legend()
    #         sb.move_legend(g, 'right')
    #
    #     else:
    #         plt.figure(figsize=(10, 5))
    #         ax = sb.stripplot(data=self.forests_and_losses, y='client_id', x=loss_type, hue='owners', palette='tab20')
    #         ax.set(title='Loss for trees per client')
    #         plt.xticks(rotation=90)
    #         ax.legend(title='Owners', loc='upper left', bbox_to_anchor=(1, 1))

    def show_prediction_errors_per_cluster(self, clustering_result, prep):
        global_forest = self.trees_and_losses.drop_duplicates(['tree'])

        for i, cluster in enumerate(clustering_result):
            cluster_df = pd.DataFrame()

            for client in cluster:
                x = prep.features_dict[client]['test'][0]
                preds_global = [tree.predict(x) for tree in global_forest.tree.to_list()]

                pred_df = pd.DataFrame(data=[np.mean(preds_global, axis=0), prep.features_dict[client]['test'][1]],
                                       columns=x.index, index=['pred_global', 'true']).T
                pred_df['pred_error'] = (pred_df.true - pred_df.pred_global)
                cluster_df[client] = pred_df.pred_error

            cluster_df['dayofweek'] = cluster_df.reset_index().timestamp.dt.dayofweek.values
            cluster_df['time'] = cluster_df.reset_index().timestamp.dt.time.values

            grouped = cluster_df.groupby(['dayofweek', 'time']).median()
            grouped = grouped.reset_index(drop=True)
            fig = px.line(data_frame=grouped,
                          title=f'Cluster {i}',
                          labels={'variable': 'client'},
                          template='plotly_white')
            fig.show()

    @staticmethod
    def show_error_scores_per_cluster(clustering_result, samples):
        for i, cluster in enumerate(clustering_result):
            cluster_clients = samples.loc[cluster].T

            fig = px.line(data_frame=cluster_clients,
                          title=f'Cluster {i}',
                          labels={'variable': 'client'},
                          template='plotly_white')
            fig.show()

    @staticmethod
    def show_consumption_per_cluster(clustering_result, prep):
        for i, cluster in enumerate(clustering_result):
            val_sets = pd.concat([prep.features_dict[client]['test'][1] for client in cluster], axis=1)
            val_sets = val_sets.astype(float)
            val_sets.columns = cluster
            val_sets['dayofweek'] = val_sets.reset_index().timestamp.dt.dayofweek.values
            val_sets['time'] = val_sets.reset_index().timestamp.dt.time.values

            grouped = val_sets.groupby(['dayofweek', 'time']).median()
            grouped = grouped.reset_index(drop=True)

            fig = px.line(data_frame=grouped,
                          title=f'Cluster {i}',
                          labels={'variable': 'client'},
                          template='plotly_white')
            fig.show()

    def get_evaluation_vectors(self, evaluation_type='global_forest', remove_workers=None):
        """ Get evaluation vector per worker. Can either be all trees in global forest or all local forests"""
        if remove_workers is None:
            remove_workers = []
        if evaluation_type == 'global_forest':
            df = self.trees_and_losses
        elif evaluation_type == 'local_forests':
            df = self.forests_and_losses
        else:
            df = None

        df = df[~df.owners.isin(remove_workers)]
        df = df[~df.client_id.isin(remove_workers)]

        ev_vectors = df.pivot_table(index='client_id',
                                    columns=['tree_num'] if evaluation_type == 'global_forest' else ['owners'],
                                    values='mse_loss')
        scaler = StandardScaler()
        ev_vectors_scaled = pd.DataFrame(scaler.fit_transform(ev_vectors.T),
                                         index=ev_vectors.columns,
                                         columns=ev_vectors.index).T

        return ev_vectors, ev_vectors_scaled

    @staticmethod
    def show_distance_matrices(evaluation_vectors, evaluation_vectors_scaled):
        """ Plot distance matrices containing pairwise distance between evaluation vectors."""
        fig, axs = plt.subplots(2, 2, figsize=(25, 20))

        for i, (df, scaling) in enumerate(
                zip([evaluation_vectors, evaluation_vectors_scaled], ['raw', 'z-scored'])):
            for j, metric in enumerate(['euclidean', 'cosine']):
                dist = pd.DataFrame(pairwise_distances(df, metric=metric), index=list(df.index),
                                    columns=list(df.index))
                axs[i, j].set_title(f'{scaling}, {metric} distances')
                _ = sb.heatmap(dist, ax=axs[i, j], cmap='Blues', xticklabels=True, yticklabels=True)
                axs[i, j].tick_params(left=False, bottom=False)
                plt.rcParams['font.size'] = 10

        fig.tight_layout()


class PSO:
    def __init__(self, samples=None, topology=Star(), max_n_clusters=15, n_iterations=5, n_particles=200,
                 n_convergence=50, cluster_validation='silhouette', distance_metric='cosine', plot_iterations=True):
        self.samples = samples.copy()
        self.topology = topology
        self.max_n_clusters = max_n_clusters
        self.n_iterations = n_iterations
        self.n_convergence = n_convergence
        self.n_particles = n_particles
        self.cluster_validation = cluster_validation
        self.distance_metric = distance_metric
        self.plot_iterations = plot_iterations

        self.options = {'c1': 1.49, 'c2': 1.49, 'w': 0.72}
        self.swarm = None

        self.m = None
        self.clustering_result = None
        self.labels = None
        self.adjust_max_n_clusters()
        self.init_m()

    def adjust_max_n_clusters(self):
        self.max_n_clusters = (self.max_n_clusters
                               if self.samples.shape[0] > self.max_n_clusters
                               else self.samples.shape[0] - 1)

    def init_m(self):
        """ Initinialize candidate centroids as random sample from data. """
        self.m = self.samples.sample(self.max_n_clusters, random_state=1)  # cluster centroids

    def execute(self):
        if self.plot_iterations:
            self.fig, self.ax = plt.subplots(1, self.n_iterations, figsize=(7 * self.n_iterations, 7))

        for i in range(self.n_iterations):
            self.swarm = ps.create_swarm(n_particles=self.n_particles, dimensions=self.max_n_clusters, binary=True,
                                         clamp=None, discrete=True, options=self.options)
            self.pso_until_convergence(i)
            self.reinit_ignored_centroids()
            self.labels = self.get_partitioning(self.swarm.best_pos)
            _ = gc.collect()

    def pso_until_convergence(self, i):
        """ Peform PSO steps until convergence (max_repeats iterations without change in best_cost). """
        cost_history = []
        prev_result = None
        iteration = 0
        iterations_without_change = 0
        self.dist_to_all_centroids = cdist(self.samples, self.m, metric=self.distance_metric)
        self.swarm.pbest_cost = np.full(200, np.inf)

        while True:
            self.pso_step()
            cost_history.append(self.swarm.best_cost)
            iteration += 1
            if self.plot_iterations:
                self.plot_cost(ax=self.ax[i], cost_history=cost_history, n_iters=iteration,
                               title=f'Iteration: {iteration} | swarm.best_cost: {self.swarm.best_cost:.4f}')
            if self.swarm.best_cost == prev_result:
                iterations_without_change += 1
            else:
                iterations_without_change = 0
                prev_result = self.swarm.best_cost
            if iterations_without_change == self.n_convergence:
                # del cost_history, prev_result, iteration, iterations_without_change
                break

    def pso_step(self):
        """ Perform 1 iteration of PSO. """
        self.swarm.current_cost = self.compute_cost_opt(self.swarm.position)  # Compute current cost
        # self.swarm.pbest_cost = self.compute_cost_opt(self.swarm.pbest_pos)  # Compute pbest pos

        self.swarm.pbest_pos, self.swarm.pbest_cost = ps.compute_pbest(self.swarm)  # Update and store

        # Part 2: Update global best (dependent on topology)
        if np.min(self.swarm.pbest_cost) < self.swarm.best_cost:
            self.swarm.best_pos, self.swarm.best_cost = self.topology.compute_gbest(self.swarm)

        # Part 3: Update position and velocity matrices
        self.swarm.velocity = self.topology.compute_velocity(self.swarm)
        self.swarm.position = self.compute_position()

    def reinit_ignored_centroids(self):
        """Randomly re-choose centroids (from samples) which were not part of the best solution."""
        m_reinit = self.m.copy()
        chosen_centroids = self.m.loc[self.swarm.best_pos.astype(bool)].index
        ignored_centroids = self.m.loc[~self.swarm.best_pos.astype(bool)].index

        new_centroids = self.samples.drop(chosen_centroids).sample(len(ignored_centroids))

        mapper = {name_old: name_new for name_old, name_new in zip(ignored_centroids, new_centroids.index)}
        m_reinit.rename(mapper, inplace=True)

        m_reinit.loc[new_centroids.index] = new_centroids
        self.m = m_reinit

    def get_partitioning(self, particle):
        """ Return partitioning of samples using centroids chosen by particle."""
        # dists = cdist(samples, chosen_centroids, metric=self.distance_metric)
        dists = self.dist_to_all_centroids[:, particle.astype(bool)]
        centroid_indices = np.argmin(dists, axis=1)

        return centroid_indices

    def clustering_score(self, labeling, metric=None):
        """ Compute cluster validation score. """
        if metric == 'silhouette':
            score = silhouette_score(self.samples.values, labeling, metric=self.distance_metric)
            return score * -1
        elif metric == 'davies_bouldin':
            score = davies_bouldin_score(self.samples.values, labeling)
            return score
        elif metric == 'dunn':
            score = self.dunn_index_cosine(self.samples.values, labeling)
            return score * -1
        else:
            return None

    # def dunn_index_cosine(self, x, labels):
    #     # Compute pairwise cosine distances between points
    #     D = cdist(x, x, metric=self.distance_metric)
    #
    #     # Find all unique cluster labels
    #     unique_labels = np.unique(labels)
    #
    #     # Compute inter-cluster distances and intra-cluster diameters
    #     inter_cluster_distances = np.zeros((len(unique_labels),))
    #     intra_cluster_diameters = np.zeros((len(unique_labels),))
    #     for k in unique_labels:
    #         cluster_k = x[labels == k, :]
    #         if len(cluster_k) == 0:
    #             continue
    #         inter_cluster_distances[k] = np.max(
    #             [np.max(D[labels == k][:, labels != k]), np.max(D[labels != k][:, labels == k])])
    #         intra_cluster_diameters[k] = np.max(cdist(cluster_k, cluster_k, metric=self.distance_metric))
    #
    #     # Compute Dunn index
    #     dunn_index = np.min(inter_cluster_distances) / np.max(intra_cluster_diameters)
    #     return dunn_index

    # def compute_cost(self, x):
    #     """ Calculate partition quality of each particle. Input are positions of all particles and possible
    #         centroids to choose from."""
    #     costs = []
    #     for particle in x:
    #         if sum(particle) < 2:
    #             costs.append(np.inf)
    #         else:
    #             chosen_centroids = self.m[particle.astype(bool)]
    #             self.get_partitioning(chosen_centroids, self.samples.values)
    #             cost = self.clustering_score(metric=self.cluster_validation)
    #             costs.append(cost)
    #
    #     return np.array(costs)

    def compute_cost_opt(self, x):
        """ Calculate partition quality of each particle. Input are positions of all particles. """
        # costs = np.full(x.shape[0], np.inf)  # Pre-fill costs with infinite

        # Process only those particles that have sum of positions >= 2
        # valid_indices = np.where(np.sum(x, axis=1) >= 2)[0]

        # all_chosen_centroids = [self.m.values[x[i].astype(bool)] for i in valid_indices]
        labeling_per_particle = [self.get_partitioning(particle) if np.sum(particle) >= 2 else None for particle in x]
        cost_per_particle = [self.clustering_score(labeling, metric=self.cluster_validation) if labeling is not None else np.inf for labeling in labeling_per_particle]
        #
        # for idx in valid_indices:
        #     particle = x[idx]
        #     # chosen_centroids = self.m.values[particle.astype(bool)]
        #     self.get_partitioning(particle)
        #     cost = self.clustering_score(metric=self.cluster_validation)
        #     costs[idx] = cost  # Set the cost only for valid particles

        return np.array(cost_per_particle)

    @staticmethod
    def sigmoid(x):

        return 1 / (1 + np.exp(-x))

    def compute_position(self):
        """Update the position matrix of the swarm
        This computes the next position in a binary swarm. It compares the
        sigmoid output of the velocity-matrix and compares it with a randomly
        generated matrix.
        """
        return (np.random.random_sample(size=self.swarm.dimensions) < self.sigmoid(self.swarm.velocity)) * 1

    def plot_cost(self, ax, cost_history, n_iters, title):
        if self.plot_iterations:
            ax.plot(np.arange(n_iters) + 1, cost_history, "b", lw=2, label='Cost', marker='.', markersize=15)

            ax.set_title(title, fontsize=10)
            ax.legend(fontsize=10)
            ax.set_xlabel('Iterations', fontsize=10)
            ax.set_ylabel('Cost', fontsize=10)
            ax.tick_params(labelsize=10)
            ax.get_legend().remove()

            display(self.fig)
            clear_output(wait=True)

    def show_clusters(self, title=None):
        df = self.samples.T.melt(ignore_index=False).reset_index().set_index('client_id')
        color_dict = dict(zip(self.samples.index, self.labels))

        df['label'] = df.index.map(color_dict)
        fig = px.line(df.reset_index().sort_values(self.samples.columns.name),
                      y='value',
                      color='label',
                      x=self.samples.columns.name,
                      line_group='client_id',
                      template='plotly_white',
                      title=title)
        fig.show()

    def show_consumption_per_cluster(self, cluster_labels, prep, per_cluster=True):
        if per_cluster:
            for label in np.unique(cluster_labels):
                cluster_clients = self.samples.index[cluster_labels == label]

                val_sets = pd.concat([prep.features_dict[client]['test'][1] for client in cluster_clients], axis=1)
                val_sets = val_sets.astype(float)
                val_sets.columns = cluster_clients
                val_sets['dayofweek'] = val_sets.reset_index().timestamp.dt.dayofweek.values
                val_sets['time'] = val_sets.reset_index().timestamp.dt.time.values

                grouped = val_sets.groupby(['dayofweek', 'time']).median()
                grouped = grouped.reset_index(drop=True)

                fig = px.line(data_frame=grouped,
                              title=f'Cluster {label}',
                              labels={'variable': 'client'},
                              template='plotly_white')
                fig.show()
        else:
            val_sets = pd.concat([prep.features_dict[client]['test'][1] for client in self.samples.index], axis=1)
            val_sets = val_sets.astype(float)
            val_sets.columns = self.samples.index
            val_sets['dayofweek'] = val_sets.reset_index().timestamp.dt.dayofweek.values
            val_sets['time'] = val_sets.reset_index().timestamp.dt.time.values

            grouped = val_sets.groupby(['dayofweek', 'time']).median()
            grouped = grouped.reset_index(drop=True)
            df = grouped.T.melt(ignore_index=False).reset_index().set_index('client_id')
            color_dict = dict(zip(self.samples.index, self.labels))

            df['label'] = df.index.map(color_dict)
            fig = px.line(df.reset_index(), y='value', color='label',
                          line_group='client_id', template='plotly_white')
            fig.show()

    def get_clusters(self):
        df = self.samples
        df['labels'] = self.labels

        return df.reset_index().groupby('labels').client_id.agg(list)

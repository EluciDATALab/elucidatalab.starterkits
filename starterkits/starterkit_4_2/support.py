import random
import zipfile
import os
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import pyswarms.backend as ps
from pyswarms.backend.topology import Star

from tqdm.notebook import tqdm
from starterkits import pipeline
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, root_mean_squared_error, davies_bouldin_score, silhouette_score
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

        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

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

        self.x_train, self.y_train = self.extract_x_y('train')
        self.x_test, self.y_test = self.extract_x_y('test')


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
        self.local_forests = None
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
        self.local_forests = []
        for nth_client, client in tqdm(enumerate(self.clients),
                                       desc='Retrieving trees', unit='client', total=len(self.clients)):
            _, forest = client.grow_forest(self.num_trees_per_client, self.tree_depth, self.min_samples_per_split)
            self.local_forests.append(forest)

    def construct_global_forest(self):
        """ Each local forest donates trees (size forest / # workers) to construct 1 global model. """
        random.seed(42)
        glob_forest = np.array([
            random.sample(forest.estimators_, int(round(self.num_trees_per_client / len(self.clients), 0) + 1))
            for forest in self.local_forests])
        glob_forest = glob_forest.flatten()
        glob_forest = glob_forest[random.sample(range(glob_forest.shape[0]), self.num_trees_per_client)]
        self.global_forest = glob_forest

    def train(self):
        """ Retrieve local forests and create global forest. """
        self.train_local_forests()
        self.construct_global_forest()

    # def evaluate_local_forests(self):
    #     """ Evaluate each local forest by each worker to create evaluation vector. """
    #     evaluations = [client.evaluate_forests(self.local_forests) for client in tqdm(self.clients)]
    #     self.forests_and_losses = pd.concat(evaluations, ignore_index=True)

    def evaluate_global_forest(self):
        """ Evaluate each tree in global forest to create evaluation vector. """
        evaluations = [client.evaluate_trees(self.global_forest) for client in tqdm(self.clients,
                                                                                    desc='Evaluating trees',
                                                                                    unit='client')]
        self.trees_and_losses = pd.concat(evaluations, ignore_index=True)

    # def show_rmse_score(self, annotate=True):
    #     """ Bar plot of MSE scores."""
    #     df = self.scores.melt(ignore_index=False, var_name='Forest type', value_name='RMSE')
    #     plt.figure(figsize=(25, 5))
    #     ax = sb.barplot(data=df, x=df.index, y='RMSE', hue='Forest type')
    #     ax.set(title='RMSE scores per client for local, cluster and global forests.')
    #     if annotate:
    #         for i in ax.containers:
    #             ax.bar_label(i, fmt="%.4f", padding=2, fontsize=8)

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

    # def show_prediction_errors_per_cluster(self, clustering_result, prep):
    #     global_forest = self.trees_and_losses.drop_duplicates(['tree'])
    #
    #     for i, cluster in enumerate(clustering_result):
    #         cluster_df = pd.DataFrame()
    #
    #         for client in cluster:
    #             x = prep.features_dict[client]['test'][0]
    #             preds_global = [tree.predict(x) for tree in global_forest.tree.to_list()]
    #
    #             pred_df = pd.DataFrame(data=[np.mean(preds_global, axis=0), prep.features_dict[client]['test'][1]],
    #                                    columns=x.index, index=['pred_global', 'true']).T
    #             pred_df['pred_error'] = (pred_df.true - pred_df.pred_global)
    #             cluster_df[client] = pred_df.pred_error
    #
    #         cluster_df['dayofweek'] = cluster_df.reset_index().timestamp.dt.dayofweek.values
    #         cluster_df['time'] = cluster_df.reset_index().timestamp.dt.time.values
    #
    #         grouped = cluster_df.groupby(['dayofweek', 'time']).median()
    #         grouped = grouped.reset_index(drop=True)
    #         fig = px.line(data_frame=grouped,
    #                       title=f'Cluster {i}',
    #                       labels={'variable': 'client'},
    #                       template='plotly_white')
    #         fig.show()
    #
    # @staticmethod
    # def show_error_scores_per_cluster(clustering_result, samples):
    #     for i, cluster in enumerate(clustering_result):
    #         cluster_clients = samples.loc[cluster].T
    #
    #         fig = px.line(data_frame=cluster_clients,
    #                       title=f'Cluster {i}',
    #                       labels={'variable': 'client'},
    #                       template='plotly_white')
    #         fig.show()
    #
    # @staticmethod
    # def show_consumption_per_cluster(clustering_result, prep):
    #     for i, cluster in enumerate(clustering_result):
    #         val_sets = pd.concat([prep.features_dict[client]['test'][1] for client in cluster], axis=1)
    #         val_sets = val_sets.astype(float)
    #         val_sets.columns = cluster
    #         val_sets['dayofweek'] = val_sets.reset_index().timestamp.dt.dayofweek.values
    #         val_sets['time'] = val_sets.reset_index().timestamp.dt.time.values
    #
    #         grouped = val_sets.groupby(['dayofweek', 'time']).median()
    #         grouped = grouped.reset_index(drop=True)
    #
    #         fig = px.line(data_frame=grouped,
    #                       title=f'Cluster {i}',
    #                       labels={'variable': 'client'},
    #                       template='plotly_white')
    #         fig.show()

    def get_evaluation_vectors(self):
        """ Get evaluation vector per worker. Can either be all trees in global forest or all local forests"""
        df = self.trees_and_losses

        ev_vectors = df.pivot_table(index='client_id',
                                    columns=['tree_num'],
                                    values='mse_loss')
        scaler = StandardScaler()
        ev_vectors_scaled = pd.DataFrame(scaler.fit_transform(ev_vectors.T),
                                         index=ev_vectors.columns,
                                         columns=ev_vectors.index).T

        return ev_vectors, ev_vectors_scaled


def forest_from_tree_list(tree_list):
    """ Create forest from list of trees. """
    forest = RandomForestRegressor(max_depth=tree_list[0].max_depth,
                                   min_samples_split=tree_list[0].min_samples_split,
                                   max_samples=0.8,
                                   # random_state=1,
                                   )
    forest.estimators_ = tree_list
    forest.n_outputs_ = 1

    forest.owners = sorted(np.unique([tree.owner for tree in tree_list]))
    forest.n_estimators = len(forest.estimators_)

    return forest

    # @staticmethod
    # def show_distance_matrices(evaluation_vectors, evaluation_vectors_scaled):
    #     """ Plot distance matrices containing pairwise distance between evaluation vectors."""
    #     fig, axs = plt.subplots(2, 2, figsize=(25, 20))
    #
    #     for i, (df, scaling) in enumerate(
    #             zip([evaluation_vectors, evaluation_vectors_scaled], ['raw', 'z-scored'])):
    #         for j, metric in enumerate(['euclidean', 'cosine']):
    #             dist = pd.DataFrame(pairwise_distances(df, metric=metric), index=list(df.index),
    #                                 columns=list(df.index))
    #             axs[i, j].set_title(f'{scaling}, {metric} distances')
    #             _ = sb.heatmap(dist, ax=axs[i, j], cmap='Blues', xticklabels=True, yticklabels=True)
    #             axs[i, j].tick_params(left=False, bottom=False)
    #             plt.rcParams['font.size'] = 10
    #
    #     fig.tight_layout()


class PSO:
    def __init__(self, samples=None, topology=Star(), max_n_clusters=10, n_iterations=5, n_particles=100,
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
        self.dist_to_candidate_centroids = None
        self.fig = None
        self.ax = None
        self.adjust_max_n_clusters()
        self.init_m()
        self.pairwise_dist = cdist(self.samples.values, self.samples.values, metric=self.distance_metric)

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
            self.dist_to_candidate_centroids = cdist(self.samples, self.m, metric=self.distance_metric)
            self.pso_until_convergence(i)
            self.reinit_ignored_centroids()
            self.labels = self.get_partitioning(self.swarm.best_pos)

    def pso_until_convergence(self, i):
        """ Peform PSO steps until convergence (max_repeats iterations without change in best_cost). """
        cost_history = []
        prev_best_cost = None
        iteration = 0
        stable_iterations = 0
        self.swarm.pbest_cost = np.full(100, np.inf)

        while True:
            self.pso_step()
            cost_history.append(self.swarm.best_cost)
            iteration += 1

            if self.plot_iterations:
                self.plot_cost(ax=self.ax[i], cost_history=cost_history, n_iters=iteration,
                               title=f'Iteration: {iteration} | swarm.best_cost: {self.swarm.best_cost:.4f}')
            if self.swarm.best_cost == prev_best_cost:
                stable_iterations += 1
            else:
                stable_iterations = 0
                prev_best_cost = self.swarm.best_cost
            if stable_iterations == self.n_convergence:
                break

    def pso_step(self):
        """ Perform 1 iteration of PSO. """
        self.swarm.current_cost = self.compute_cost(self.swarm.position)  # Compute current cost
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
        dists_to_chosen_centroids = self.dist_to_candidate_centroids[:, particle.astype(bool)]
        clustering = np.argmin(dists_to_chosen_centroids, axis=1)

        return clustering

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

    def dunn_index_cosine(self, x, labels):
        # Compute pairwise cosine distances between points
        # Find all unique cluster labels
        unique_labels = np.unique(labels)

        # Compute inter-cluster distances and intra-cluster diameters
        inter_cluster_distances = np.zeros((len(unique_labels),))
        intra_cluster_diameters = np.zeros((len(unique_labels),))
        for k in unique_labels:
            cluster_k = x[labels == k, :]
            if len(cluster_k) == 0:
                continue
            inter_cluster_distances[k] = np.max(
                [np.max(self.pairwise_dist[labels == k][:, labels != k]), np.max(self.pairwise_dist[labels != k][:, labels == k])])
            intra_cluster_diameters[k] = np.max(cdist(cluster_k, cluster_k, metric=self.distance_metric))

        # Compute Dunn index
        dunn_index = np.min(inter_cluster_distances) / np.max(intra_cluster_diameters)
        return dunn_index

    def compute_cost(self, x):
        """ Calculate partition quality of each particle. Input are positions of all particles. """
        labeling_per_particle = [self.get_partitioning(particle) if np.sum(particle) >= 2 else None for particle in x]
        cost_per_particle = [self.clustering_score(labeling, metric=self.cluster_validation)
                             if labeling is not None else np.inf for labeling in labeling_per_particle]

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


class FedRepo:
    def __init__(self, data=None, forest_support_threshold=0.0333, worker_repo_threshold=0.2, num_trees=100):
        self.local_forests = {}
        self.active_models = {}
        self.thresh_dict = {}
        self.worker_data = data
        self.client_ids = data.consumer_list
        self.evaluation_vectors = None
        self.clusters = None
        self.global_forest = None

        self.forest_support_threshold = forest_support_threshold
        self.worker_repo_threshold = int(worker_repo_threshold * len(self.client_ids))
        self.num_trees = num_trees

        self.worker_repo = []
        self.tree_repo = {}
        self.maintenance_log = {}
        self.worker_repository_hist = []
        self.retrain_needed = False
        self.predicted_day = None
        self.days_since_retrain = None

        self.without_maintenance_dict = {worker: {} for worker in self.client_ids}
        self.with_maintenance_dict = {worker: {'predictions': [], 'rmse': [],
                                               'pred_index': [], 'active_model': []} for worker in self.client_ids}
        # random.seed(1)

    def initialize(self):
        """ Initialization stage. """
        self.worker_repo = self.client_ids
        self.active_models = {}

    def training(self):
        """ Training stage. """
        fed_forest = FederatedForest(client_ids=self.worker_repo)
        fed_forest.assign_client_data(self.worker_data.x_train, self.worker_data.y_train,
                                      self.worker_data.x_test, self.worker_data.y_test)

        fed_forest.train()
        fed_forest.evaluate_global_forest()
        _, self.evaluation_vectors = fed_forest.get_evaluation_vectors()

        for forest in fed_forest.local_forests:
            self.local_forests[forest.owners] = forest

        self.update_tree_repo()
        self.set_global_model()

    def clustering(self, plot_iterations=True):
        """ Clustering stage. """
        pso_clusterer = PSO(samples=self.evaluation_vectors, cluster_validation='dunn', plot_iterations=plot_iterations)
        pso_clusterer.execute()
        self.clusters = pso_clusterer.get_clusters()
        self.create_cluster_models()
        self.determine_thresholds()

        self.worker_repo = []

    def concept_drift_mitigation(self):
        """ Generate predictions and monitor performance for validation set. Retrain if needed. """
        days_val_set = pd.date_range(start='2012-08-01', end='2013-12-31', freq='D')

        self.days_since_retrain = 0

        for day in tqdm(days_val_set, desc='Days predicted', unit='day'):
            self.predicted_day = day
            self.generate_cluster_predictions(day)

            if self.days_since_retrain >= 3:
                self.check_rmse_thresholds()

            self.set_model_support()
            self.check_support()
            self.check_worker_allocation()
            self.worker_repository_hist.append(len(self.worker_repo))

            self.check_worker_repo()
            self.days_since_retrain += 1
            self.clean_models()

            if self.retrain_needed:
                print(f'Worker repository contains {len(self.worker_repo)} workers. Retraining is invoked.')
                self.retrain()
                self.retrain_needed = False
                self.days_since_retrain = 0

    def retrain(self):
        """ Retraining stage. """
        end_date = self.predicted_day
        start_date_train = self.predicted_day - timedelta(days=120)
        end_date_train = self.predicted_day - timedelta(days=30)

        x_train = []
        y_train = []
        x_test = []
        y_test = []

        for worker in self.worker_repo:
            x_train.append(self.worker_data.features_dict[worker]['val'][0][start_date_train: end_date_train])
            y_train.append(self.worker_data.features_dict[worker]['val'][1][start_date_train: end_date_train])

            x_test.append(self.worker_data.features_dict[worker]['val'][0][end_date_train: end_date])
            y_test.append(self.worker_data.features_dict[worker]['val'][1][end_date_train: end_date])

        # delete old global forest
        del self.active_models[self.global_forest]
        self.maintenance_log[self.predicted_day] = self.worker_repo

        self.training()
        print('Finding optimal clustering...')
        self.clustering(plot_iterations=False)

    def update_tree_repo(self):
        """ Donate trees from local forests to tree repo. """
        portion = 5

        for worker in self.worker_repo:
            self.tree_repo[worker] = random.sample(self.local_forests[worker].estimators_, int(portion))

    def set_global_model(self):
        """ Set/update global model. """
        tree_pool = np.array([trees for worker, trees in self.tree_repo.items()]).flatten()
        selected_trees = random.sample(population=list(tree_pool), k=self.num_trees)

        self.global_forest = forest_from_tree_list(selected_trees)
        self.global_forest.support = 0
        self.active_models[self.global_forest] = []

    def create_cluster_models(self):
        """ Create federated model for each cluster. """
        for cluster in self.clusters:
            tree_pool = np.array([self.local_forests[worker].estimators_ for worker in cluster]).flatten()
            selected_trees = random.sample(population=list(tree_pool), k=self.num_trees)
            cluster_forest = forest_from_tree_list(selected_trees)

            self.active_models[cluster_forest] = cluster.copy()

        self.active_models[self.global_forest] = []
        self.set_model_support()

    def get_model(self, worker):
        """ Return key (model) of worker (in worker list). """
        return [key for key, worker_list in self.active_models.items() if worker in worker_list][0]

    def clean_models(self):
        """ Clean up any cluster model that has been deactivated. """
        self.active_models = {model: self.active_models[model] for model in self.active_models
                              if len(self.active_models[model]) != 0 or model == self.global_forest}

    def predict_test_set(self):
        """ Generate predictions using cluster model on test set. """
        pred_dict = {}

        for worker in self.worker_repo:
            feature_df, target = self.worker_data.features_dict[worker]['test']
            cluster_forest = self.get_model(worker)
            preds_cluster = cluster_forest.predict(feature_df)
            pred_df = pd.DataFrame({'pred_cluster': preds_cluster,
                                    'true': target},
                                   index=feature_df.index)

            pred_dict[worker] = pred_df

        return pred_dict

    def get_daily_rmse_test(self):
        """ Get daily RMSE for predictions of cluster forest for each worker's test set. """
        rmse_dict = {}
        pred_dict_test = self.predict_test_set()

        for worker in pred_dict_test:
            pred_df = pred_dict_test[worker].copy()
            pred_df['date'] = pred_df.index.date  # Assuming 'index' is of datetime type

            daily_rmses = []

            for date, day in pred_df.groupby('date'):
                cluster_rmse = np.sqrt(mean_squared_error(day['true'], day['pred_cluster']))
                daily_rmses.append([date, cluster_rmse])

            rmse_dict[worker] = pd.DataFrame(daily_rmses, columns=['date', 'cluster']).set_index('date')

        return rmse_dict

    def determine_thresholds(self):
        """ Determine thresholds for concept drift based on daily RMSE on training set of cluster forest. """
        rmse_dict = self.get_daily_rmse_test()
        for worker in rmse_dict:
            thresh = rmse_dict[worker].cluster.mean() + (3 * rmse_dict[worker].cluster.std())
            self.thresh_dict[worker] = thresh

    def check_rmse_thresholds(self):
        """ Check RMSE of past three days for every worker and compare with threshold. """
        for worker in self.client_ids:
            if worker not in set(self.worker_repo):
                rmse_hist = self.with_maintenance_dict[worker]['rmse'][-3:]
                exceeds_threshold = all(score > self.thresh_dict[worker] for score in rmse_hist)
                if exceeds_threshold:
                    print(f'{self.predicted_day.date()}: global model is activated for {worker}.')
                    self.activate_global_model(worker)

    def activate_global_model(self, worker):
        """ Remove worker from active cluster forest, activate global model and add worker to maintenance. """
        self.worker_repo.append(worker)

        cluster_forest = self.get_model(worker)
        self.active_models[cluster_forest].remove(worker)
        self.active_models[self.global_forest].append(worker)

    def set_model_support(self):
        """ Divide amount of workers predicted by active forest by total amount to get model support. """
        for (active_model, workers) in self.active_models.items():
            active_model.support = len(workers) / len(self.client_ids)

    def check_support(self):
        """ Check support of federated models. If below threshold, deactivate model and replace by global forest. """
        for active_model in self.active_models.keys():
            if (active_model.support <= self.forest_support_threshold) & (active_model != self.global_forest):
                print(f'{self.predicted_day.date()}: Model support too low. Global model activated '
                      f'for workers {self.active_models[active_model]}.')
                self.worker_repo.extend(self.active_models[active_model])
                self.active_models[self.global_forest].extend(self.active_models[active_model])
                self.active_models[active_model] = []

    def check_worker_repo(self):
        """ Check if maintenance repo exceeds limit. If so, retrain is needed."""
        if len(self.worker_repo) >= self.worker_repo_threshold:
            self.retrain_needed = True

    def generate_cluster_predictions(self, day):
        """ Generate predictions for each worker using their active model. """
        for cluster_forest, cluster in self.active_models.items():
            for worker in cluster:
                features_day = self.worker_data.features_dict[worker]['val'][0].loc[day.strftime('%Y-%m-%d')]
                target = self.worker_data.features_dict[worker]['val'][1].loc[day.strftime('%Y-%m-%d')]

                if features_day.empty:
                    continue

                preds = cluster_forest.predict(features_day)
                rmse = root_mean_squared_error(target, preds)

                self.with_maintenance_dict[worker]['rmse'].append(rmse)
                self.with_maintenance_dict[worker]['pred_index'].extend(features_day.index)
                self.with_maintenance_dict[worker]['predictions'].extend(preds)
                self.with_maintenance_dict[worker]['active_model'].append('global' if worker in self.worker_repo
                                                                          else 'cluster')

    def check_worker_allocation(self):
        """ Ensure all workers are accounted for in the active models. """
        workers_with_models = sum(len(cluster) for cluster in self.active_models.values())
        assert workers_with_models == len(self.client_ids)

    def generate_without_maintenance(self):
        """ Generate predictions for each worker using their active model. """
        self.without_maintenance_dict = {worker: {} for worker in self.client_ids}

        for worker in tqdm(self.client_ids, desc='Generating predictions', unit='worker'):
            cluster_forest = self.get_model(worker)
            features = self.worker_data.features_dict[worker]['val'][0]
            target = self.worker_data.features_dict[worker]['val'][1]

            preds_cluster = cluster_forest.predict(features)
            preds_local = self.local_forests[worker].predict(features)
            preds_global = self.global_forest.predict(features)

            self.without_maintenance_dict[worker]['pred_cluster'] = preds_cluster
            self.without_maintenance_dict[worker]['pred_local'] = preds_local
            self.without_maintenance_dict[worker]['pred_global'] = preds_global
            self.without_maintenance_dict[worker]['true'] = target

    # def plot_rmse(self, rmse_dict):
    #     """ Plot RMSE per forest type and worker over full predicted set. """
    #     df = pd.DataFrame(rmse_dict, index=['local', 'cluster', 'global']).T
    #     df['label'] = self.labels
    #     melted = df.melt(id_vars='label', var_name='Forest type', value_name='RMSE', ignore_index=False).reset_index()
    #
    #     ax = sb.catplot(data=melted, kind='bar', x='index', row='label', y='RMSE', hue='Forest type', height=6,
    #                     aspect=3, sharex=False)
    #     ax.set_xticklabels(rotation=45)
    #     ax.tight_layout()
    #
    # def plot_rmse_line(self, rmse_dict):
    #     # df = pd.DataFrame(rmse_dict, index=['local', 'global', 'active', 'without_maintenance']).T
    #     df = pd.DataFrame(rmse_dict, index=['local', 'federated', 'global']).T
    #
    #     df['cluster'] = self.labels
    #     df = df.drop('MAC004863')
    #     df = df.melt(id_vars='cluster', ignore_index=False)
    #     fig = px.line(data_frame=df, y='value', line_dash='variable', template='plotly_white', facet_row='cluster', color='cluster', height=500, line_dash_sequence=['dot', 'solid', 'dash'], width=900, facet_row_spacing=0.14,
    #                   labels={'variable': 'Forest type', 'value': 'RMSE'}
    #                   )
    #     fig.update_layout(font=dict(size=14))
    #     fig.update_xaxes(matches=None, mirror='all', automargin=True)
    #     # fig.update_yaxes(matches=None)
    #     fig.for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True))
    #     fig.update_layout(showlegend=False)
    #     fig.for_each_annotation(lambda a: a.update(text=''))
    #     # fig.update_layout(xaxes=dict(tickfont=dict(size=2)))
    #     fig.update_layout(autosize=False, width=1000, height=1200,)
    #     fig.update_yaxes(matches=None)
    #     fig.update_xaxes(tickfont=dict(size=7))
    #
    #     # fig.update_layout(legend=dict(title='Model',
    #     #                     orientation="h",
    #     #                     entrywidth=50,
    #     #                     yanchor="bottom",
    #     #                     y=1.02,
    #     #                     xanchor="right",
    #     #                     x=1
    #     #                 ))
    #     fig.for_each_xaxis(lambda xaxis: xaxis.update(showticklabels=True, tickangle=45, title='household'))
    #
    #     fig.show()

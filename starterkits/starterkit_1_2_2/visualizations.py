# ©, 2022, Sirris
# owner: FFNG

"""
Utilities for visualisations.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
import re
import numpy as np
import kneed
from ipywidgets import interact, Layout
import ipywidgets as widgets
from copy import deepcopy
from matplotlib.gridspec import GridSpec
from starterkits.starterkit_1_2_2.data_handler import load_train_data, BASE_PATH_HEALTHY
from starterkits.starterkit_1_2_2.support import calculate_roc_characteristics, calc_tpr_at_fpr_threshold, \
    calc_fpr_at_tpr_threshold
import glob
import os


previous_value_threshold = 95


def get_number_of_components(explained_variance, threshold, df_nmf_models_):
    if np.max(explained_variance) < threshold:
        n_components_variance = len(explained_variance)
    else:
        n_components_variance = (explained_variance > threshold).argmax() + 1

    kneedle = kneed.KneeLocator(x=np.arange(1, len(explained_variance) + 1), y=explained_variance,
                                direction='increasing', S=0)
    n_components_knee_pca = kneedle.knee

    kneedle = kneed.KneeLocator(x=df_nmf_models_['n_components'],
                                y=df_nmf_models_['reconstruction_error'],
                                direction='decreasing', curve='convex')
    n_components_knee_nmf = kneedle.elbow

    return n_components_variance, n_components_knee_pca, n_components_knee_nmf


def _extract_lower_limit(x):
    return float(re.findall(r'(?<=\_)(.*)(?=-)', x)[0])


def _extract_upper_limit(x):
    return float(re.findall(r'(?<=-)(.*)$', x)[0])


def get_controller(controller):
    """
    Parse dictionary into widget.

    :param controller: dict. Dictionary with inputs to widget. Must have at
    least key indicating name of widget
    :return:
    widget
    """
    controller = deepcopy(controller)
    wdgt = controller['widget']

    controller.pop('widget')
    controller['continuous_update'] = controller.get('continuous_update',
                                                     False)
    controller['style'] = controller.get('style',
                                         {'description_width': 'initial'})
    controller['layout'] = Layout(width='60%')
    controller_out = getattr(widgets, wdgt)(**controller)
    return controller_out


def illustrate_nmf_components_interactive(df_V_train, df_nmf_models):
    global previous_value_threshold
    previous_value_threshold = 95

    saved_values = {}

    max_n_components = len(df_nmf_models)

    def make_plot(criterium, n_components_range, **extra_options):
        global previous_value_threshold

        if 'min_explained_variance' in extra_options.keys():
            threshold = extra_options['min_explained_variance']
        else:
            threshold = previous_value_threshold

        previous_value_threshold = threshold

        n_components_variance, n_components_knee_pca, n_components_knee_nmf = get_number_of_components(
            explained_variance_ratio,
            threshold,
            df_nmf_models)

        if criterium == 'explained-variance':
            n_components = n_components_variance
        elif criterium == 'knee-point':
            n_components = max(n_components_knee_nmf, n_components_knee_pca)
        elif criterium == 'both':
            n_components = max(n_components_variance, n_components_knee_nmf, n_components_knee_pca)
        else:
            n_components = 5

        n_components = max(n_components, n_components_range[0])
        n_components = min(n_components, n_components_range[1])

        saved_values['n_components'] = n_components

        components_per_row = 3
        n_rows_first_plot = 3
        n_rows = n_rows_first_plot + int(np.ceil(n_components / components_per_row))
        n_cols = 2 * components_per_row

        fig = plt.figure(layout="constrained", figsize=(12, n_rows))

        gs = GridSpec(n_rows, n_cols, figure=fig)
        ax1 = fig.add_subplot(gs[:n_rows_first_plot, :n_cols // 2])
        ax2 = fig.add_subplot(gs[:n_rows_first_plot, n_cols // 2:])
        # identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))

        ax1.plot(np.arange(1, max_n_components + 1), explained_variance_ratio, 'ko-')
        ax2.plot(df_nmf_models['n_components'],
                 df_nmf_models['reconstruction_error'], 'ko-')

        if criterium in ('both', 'explained-variance'):
            ax1.axhline(y=extra_options['min_explained_variance'], linestyle=':', color='g')
            ax1.axvline(x=n_components_variance, linestyle=':', color='g',
                        label=f'Explained variance PCA (N={n_components_variance})')
        if criterium in ('both', 'knee-point'):
            ax1.axvline(x=n_components_knee_pca, linestyle=':', color='b',
                        label=f'Knee point PCA (N={n_components_knee_pca})')
            ax2.axvline(x=n_components_knee_nmf, linestyle=':', color='r',
                        label=f'Knee point NMF (N={n_components_knee_nmf})')
            ax2.legend(loc='upper right', fontsize='small')
        ax1.set_xlabel("Number of components")
        ax1.set_ylabel("Cumulative explained variance")
        ax2.set_xlabel("Number of components")
        ax2.set_ylabel("Frobenium norm")

        ax1.legend(loc='lower right', fontsize='small')

        row = df_nmf_models[df_nmf_models.n_components == n_components].iloc[0]
        H = row.H

        axes_components = []

        column_names = df_V_train.columns[0]
        offset = 0.5 * (_extract_upper_limit(column_names) - _extract_lower_limit(column_names))
        all_x_label_names = [_extract_lower_limit(col) + offset for col in df_V_train.columns]
        plot_x_ticks = [0, 10, 20, 30, 40, 49]
        plot_x_labels = [all_x_label_names[idx] for idx in plot_x_ticks]

        for _i_component in range(n_components):
            i_row = n_rows_first_plot + (_i_component // components_per_row)
            i_col = 2 * (_i_component % components_per_row)

            if len(axes_components) > 0:
                _ax = fig.add_subplot(gs[i_row, i_col:i_col + 2],
                                      sharey=axes_components[0], sharex=axes_components[0])
            else:
                _ax = fig.add_subplot(gs[i_row, i_col:i_col + 2])

            _ax.plot(np.arange(H.shape[1]), H.iloc[_i_component])
            _ax.text(0, 2.5, f'Component {_i_component + 1}', fontdict={'size': 10})
            _ax.set_xticks(plot_x_ticks)
            _ax.set_xticklabels(plot_x_labels)

            axes_components.append(_ax)
            if i_row == (n_rows - 1):
                _ax.set_xlabel('Frequency [orders]')

        fig.show()

    def get_additional_options(criterium, n_components_range):
        global previous_value_threshold

        threshold = 95 if previous_value_threshold is None else previous_value_threshold
        if criterium in ('explained-variance', 'both'):
            controller_explained_variance = get_controller({'widget': 'FloatSlider',
                                                            'min': min_variance,
                                                            'max': 100,
                                                            'value': threshold,
                                                            'description': 'Explained PCA variance threshold'})
            extra_options = {'min_explained_variance': controller_explained_variance}
        else:
            extra_options = {}

        def _make_plot(**_extra_options):
            make_plot(criterium, n_components_range, **_extra_options)

        interact(_make_plot, **extra_options)

    # calculate explained variance
    pca = PCA(n_components=max_n_components, random_state=42)
    pca.fit(df_V_train)
    explained_variance_ratio = pca.explained_variance_ratio_.cumsum()
    explained_variance_ratio = explained_variance_ratio[:max_n_components] * 100
    min_variance = np.ceil(np.min(explained_variance_ratio))

    controller_selection_criterium = get_controller({'widget': 'Dropdown',
                                                     'options': [('Explained variance', 'explained-variance'),
                                                                 ('Knee point', 'knee-point'),
                                                                 ('Both', 'both')],
                                                     'value': 'explained-variance',
                                                     'description': 'How to select the number of components for NMF'})

    controller_n_components_range = get_controller({'widget': 'IntRangeSlider',
                                                    'min': 1,
                                                    'max': max_n_components,
                                                    'value': [2, 10],
                                                    'description': 'Range of acceptable number of components for NMF'})

    interact(get_additional_options, criterium=controller_selection_criterium,
             n_components_range=controller_n_components_range)

    return saved_values


def show_fingerprints(model, df_V_train, meta_data_train, df_operating_modes):

    def make_plot(rpm, torque, run):
        _n_components = df_W.shape[1] - 5

        fontsize = max(10, _n_components * 0.75)
        fontsize_hm = max(10, _n_components * 0.4)

        fig, ax = plt.subplots(figsize=(_n_components, 4))
        if run == 'Mean over all measurements':
            df_ = df_W[(df_W['rotational speed [RPM]'] == rpm) & (df_W['torque [Nm]'] == torque)]
            df_ = df_[list(range(_n_components)) + ['direction']].groupby('direction').mean()
            # fig, ax = plt.subplots()
            sns.heatmap(df_, annot=True, fmt=".3f", ax=ax, cmap='Blues', vmin=0, vmax=0.1, cbar=False,
                        annot_kws={"size": fontsize_hm})
            ax.set_title(f'Vibration fingerprint @ {rpm} rpm, {torque} Nm')
            ax.set_xlabel('component')
        else:
            df_ = df_W[(df_W['rotational speed [RPM]'] == rpm) &
                       (df_W['torque [Nm]'] == torque) &
                       (df_W['sample_id'] == run)]
            df_ = df_.set_index('direction')

            sns.heatmap(df_[list(range(_n_components))], annot=True, fmt=".3f", ax=ax, cmap='Blues', vmin=0, vmax=0.1,
                        cbar=False, annot_kws={"size": fontsize_hm})
            ax.set_title(f'Measurement {run} @ {rpm} rpm, {torque} Nm')
            ax.set_xlabel('component')

        # Modify the y-axis tick labels to remove the first 4 letters
        y_ticks = ax.get_yticklabels()
        new_labels = [label.get_text()[4:] for label in y_ticks]
        ax.set_yticklabels(new_labels)

        ax.tick_params(axis='x', labelsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize)

        ax.title.set_fontsize(fontsize)
        ax.xaxis.label.set_fontsize(fontsize)
        ax.yaxis.label.set_fontsize(fontsize)

        fig.show()

    n_components = model.W.shape[1]
    W_train = model.W.reshape(-1, n_components)
    df_W_train = pd.DataFrame(W_train)
    df_W_train.index = df_V_train.index
    df_W_train['direction'] = meta_data_train['direction']

    # add operating mode (OM)
    df_W = pd.merge(df_W_train, meta_data_train.drop(columns=['direction']), left_index=True, right_index=True)

    possible_oms = df_operating_modes.values.flatten()
    possible_oms = np.delete(possible_oms, np.where(possible_oms == ''))

    controller_om = get_controller({'widget': 'Dropdown',
                                    'options': [(_om, _om) for _om in possible_oms],
                                    'value': possible_oms[0],
                                    'description': 'Select the operating mode'})

    def get_additional_options(operating_mode):
        idx, col = np.where(df_operating_modes == operating_mode)
        torque = df_operating_modes.index[idx][0]
        rpm = df_operating_modes.columns[col][0]
        df_selected = df_W[(df_W['rotational speed [RPM]'] == rpm) & (df_W['torque [Nm]'] == torque)]
        possible_runs = ['Mean over all measurements'] + sorted(df_selected['sample_id'].unique())

        controller_run = get_controller({'widget': 'Dropdown',
                                         'options': [(_run, _run) for _run in possible_runs],
                                         'value': possible_runs[0],
                                         'description': 'Select the measurement'})

        def _make_plot(run):
            make_plot(rpm, torque, run)
        interact(_make_plot, run=controller_run)

    interact(get_additional_options, operating_mode=controller_om)


def plot_example_interactive():
    def make_plot(rpm_torque, _run, _run_to_idx):
        _rpm = int(rpm_torque.split(' - ')[0].strip('RPM'))
        _torque = int(rpm_torque.split(' - ')[1].strip('Torque'))
        run_idx = _run_to_idx[_run]
        df_example = load_train_data(_rpm, _torque, _run)
        print(f"A single vibration measurement (rpm={_rpm}, torque={_torque}, run={run_idx}) has the following shape: "
              f"{df_example.shape}")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for var, ax in zip(['x', 'y', 'z'], axes):
            ax.plot(df_example[var], label=var)
            ax.set_title(var)
            ax.legend()
        fig.show()

    file_names_healthy = glob.glob(os.path.join(BASE_PATH_HEALTHY, '*.txt'))
    rpms = [int(_f.split('/')[-1].split('_')[0].strip('V')) for _f in file_names_healthy]
    torques = [int(_f.split('/')[-1].split('_')[1].strip('N')) for _f in file_names_healthy]
    runs_per_combination = {}
    for rpm in sorted(np.unique(rpms)):
        for torque in sorted(np.unique(torques)):
            runs = glob.glob(os.path.join(BASE_PATH_HEALTHY, f'V{rpm}_{torque}N_*.txt'))
            runs = [int(run.split('/')[-1].split('_')[2].strip('.txt')) for run in runs]
            if len(runs) > 0:
                runs_per_combination[f'RMP {rpm} - Torque {torque}'] = runs

    possible_combinations = list(runs_per_combination.keys())

    controller_rpm_torque = get_controller({'widget': 'Dropdown',
                                            'options': [(_combination, _combination) for _combination in
                                                        possible_combinations],
                                            'value': possible_combinations[0],
                                            'description': 'Chose the context'})

    def get_additional_options(rpm_torque):
        possible_runs = sorted(runs_per_combination[rpm_torque])
        run_to_idx = {_run: _idx+1 for _idx, _run in enumerate(possible_runs)}
        controller_run = get_controller({'widget': 'Dropdown',
                                         'options': [(_idx+1, _run) for _idx, _run in enumerate(possible_runs)],
                                         'value': possible_runs[0],
                                         'description': 'Which run'})

        def _make_plot(run):
            make_plot(rpm_torque, run, run_to_idx)

        interact(_make_plot, run=controller_run)

    interact(get_additional_options, rpm_torque=controller_rpm_torque)


def plot_ROC_curve(df_cosine):
    df_cosine = df_cosine[~np.isnan(df_cosine.distance_to_own_cluster_center)]  # removed unknown cluster labels

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the general ROC curve
    fpr, tpr, roc_auc = calculate_roc_characteristics(df_cosine)
    ax.plot(fpr, tpr, color='blue', lw=4, label=f'overall (area = {roc_auc:.3f})', alpha=0.66)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='baseline')
    # TPR@FPR
    threshold = 0.1
    tpr_at_fpr = calc_tpr_at_fpr_threshold(tpr, fpr, threshold=threshold)
    ax.plot([0, threshold], [tpr_at_fpr, tpr_at_fpr], color='red', lw=2, linestyle='--',
            label=f'TPR@FPR={threshold:.2f} = {tpr_at_fpr:.2f}')
    ax.plot([threshold, threshold], [0, tpr_at_fpr], color='red', lw=2, linestyle='--')
    # FPR@TPR
    threshold = 0.9
    fpr_at_tpr = calc_fpr_at_tpr_threshold(tpr, fpr, threshold=threshold)
    ax.plot([fpr_at_tpr, fpr_at_tpr], [0, threshold], color='green', lw=2, linestyle='--',
            label=f'FPR@TPR={threshold:.2f} = {fpr_at_tpr:.2f}')
    ax.plot([0, fpr_at_tpr], [threshold, threshold], color='green', lw=2, linestyle='--')
    # limits
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve')
    n_total = len(df_cosine)
    n_healthy = len(df_cosine[~df_cosine['pitting']])
    n_unhealthy = len(df_cosine[df_cosine['pitting']])
    text = f"n={n_total} ({n_healthy} healthy, {n_unhealthy} unhealthy)"
    ax.annotate(xy=(0.1, 0.025), text=text)

    ax.legend(loc='lower right', title='Pitting severity level')
    fig.show()


def plot_weights_interactive(df_W_online, meta_data_test, df_operating_modes, n_components):

    fontsize = max(10, n_components * 0.75)
    fontsize_hm = max(10, n_components * 0.4)

    def plot_weights(period):
        usid = df_W_online['unique_sample_id'][period]
        df_ = meta_data_test[meta_data_test['unique_sample_id'] == usid]
        rpm = df_['rotational speed [RPM]'].iloc[0]
        torque = df_['torque [Nm]'].iloc[0]
        operating_mode = df_operating_modes.loc[torque, rpm]

        fig, ax = plt.subplots(figsize=(n_components, 4))
        sns.heatmap(df_W_online['W'][period], annot=True, fmt=".3f", ax=ax, cmap='Blues', vmin=0, vmax=0.05, cbar=False,
                    annot_kws={"size": fontsize_hm})

        ax.set_title(
            f'Derived weights for measurement {period} @ {rpm} rpm, {torque} Nm (operating mode {operating_mode})')
        ax.set_yticklabels(['x', 'y', 'z'], rotation=0)
        ax.set_ylabel('Measurement direction')
        ax.set_xlabel('Component')

        ax.tick_params(axis='x', labelsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize)

        ax.title.set_fontsize(fontsize)
        ax.xaxis.label.set_fontsize(fontsize)
        ax.yaxis.label.set_fontsize(fontsize)

        fig.show()

    possible_periods = sorted(list(df_W_online['unique_sample_id'].index))
    controller_period = get_controller({'widget': 'Dropdown',
                                        'options': [(_period, _period) for _period in possible_periods],
                                        'value': possible_periods[0],
                                        'description': 'Select measurement'})
    interact(plot_weights, period=controller_period)

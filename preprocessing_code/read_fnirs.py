"""

  Module: read_fnirs.py

  An example of reading fNIRS signals using MNE package for the Steinmetzger_et_al datasets.

  This script processes fNIRS and associated EEG data for auditory pitch experiments using the Steinmetzger_et_al datasets.
  It supports loading raw SNIRF and MATLAB .mat files, performing quality checks, artifact correction (TD-DR, Kalman, CBSI), epoching,
  HRF modeling via GLM, and generates a suite of plots (time series, average responses, boxplots, HRF curves).

  Usage:
    python read_fnirs.py <sel_plot> <sel_method> <channel_index> <sel_load>

  Arguments:
    sel_plot (int): Flag to enable HRF GLM modeling and plotting (1) or skip (0).
    sel_method (int): Artifact correction method: 0 = TD-DR, 1 = Kalman, 2 = CBSI.
    channel_index (int): Index of the channel to compute AUC and HRF responses for. Typically 34 or 36 for
    the purposes of the Steinmetzger_et_al paper.
    sel_load (int): If 0, load raw SNIRF files and process; if 1, load existing preprocessed NPZ and update if needed.

  Outputs:
    - Figures saved under '../figs_folder_auditory/' (time series, average responses, median splits, HRF curves).
    - Compressed NPZ files '../processed_data_<method>_stim_wise.npz' with all condition-wise data and indices

"""

import mne
import sys
import mne_nirs
import os
import pandas as pd
import seaborn as sns
import numpy as np
import pickle
import h5py
import matplotlib as mpl
import statsmodels.formula.api as smf
from mne.io import read_raw_snirf
from typing import Any
from pykalman import KalmanFilter
from mne import concatenate_epochs
from mne_bids import BIDSPath, read_raw_bids
from itertools import compress
from matplotlib import pyplot as plt
from loguru import logger
from os import listdir
from scipy import io
from os.path import isfile, join
from mne_nirs.experimental_design import make_first_level_design_matrix
from nilearn.plotting import plot_design_matrix
from mne_nirs.statistics import run_glm, statsmodels_to_results
from mne_nirs.visualisation import plot_glm_group_topo, plot_glm_surface_projection
from statsmodels.stats.multitest import multipletests
from nilearn.glm.first_level import glover_hrf, glover_time_derivative, glover_dispersion_derivative
# from nilearn.glm.first_level import spm_hrf, spm_hrf_derivative

colors_tuple_list = [
    (0.894, 0.102, 0.110),  # Red
    (0.216, 0.494, 0.722),  # Blue
    (0.302, 0.686, 0.290),  # Green
    (0.596, 0.306, 0.639),  # Purple
    (1.000, 0.498, 0.000),  # Orange
    (1.000, 1.000, 0.200),  # Yellow
    (0.651, 0.337, 0.157),  # Brown
    (0.969, 0.506, 0.749),  # Pink
    (0.600, 0.600, 0.600),  # Gray
    (0.737, 0.741, 0.133),  # Olive
    (0.090, 0.745, 0.811),  # Cyan
    (0.121, 0.470, 0.705),  # Dark Blue
    (0.737, 0.502, 0.741),  # Lavender
    (0.792, 0.698, 0.839),  # Light Purple
    (0.090, 0.745, 0.466),  # Teal Green
    (0.835, 0.369, 0.000),  # Burnt Orange
    (0.580, 0.000, 0.827),  # Violet
    (0.000, 0.502, 0.502),  # Dark Cyan
    (0.502, 0.000, 0.000),  # Dark Red
    (0.000, 0.392, 0.000),  # Dark Green
    (0.000, 0.000, 0.545),  # Dark Blue Navy
    (0.545, 0.000, 0.545),  # Dark Magenta
    (0.502, 0.502, 0.000),  # Olive Drab
    (0.824, 0.706, 0.549),  # Tan
    (0.251, 0.878, 0.816),   # Turquoise
    (0.863, 0.078, 0.235),  # Crimson
    (0.255, 0.412, 0.882),  # Royal Blue
    (0.133, 0.545, 0.133),  # Forest Green
    (1.000, 0.388, 0.278),  # Coral
    (0.545, 0.271, 0.075)   # Saddle Brown
]

kf = KalmanFilter(
    transition_matrices=[1],
    observation_matrices=[1],
    transition_covariance=1e-6,   # tune for smoothness
    observation_covariance=0.01    # tune for noise
)

# get here the auxiliary functions


def load_matlab_file(path):
    """
       Load a .mat file, v7 or earlier via scipy.io.loadmat,
       or v7.3+ via h5py.
       Parameters
       ----------
        path : str
           Filesystem path to the .mat file.

       Returns
       -------
         data : dict
            Mapping of variable names to NumPy arrays.

       Raises
       ------
       FileNotFoundError
          If the file does not exist at the given path.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No such file: {path}")

    # Try HDF5 open first for v7.3+ files
    try:
        with h5py.File(path, 'r') as f:
            # if this succeeds, read all datasets into numpy arrays
            return {key: f[key][()] for key in f.keys()}
    except (OSError, IOError):
        # Not an HDF5 file, fall back to scipy.loadmat
        mat = io.loadmat(path)
        # remove MATLAB meta-vars if you like
        for m in ('__header__', '__version__', '__globals__'):
            mat.pop(m, None)
        return mat


def cbs_i(hbo, hbr):
    """
    Apply Correlation-Based Signal Improvement (CBSI).

    Parameters
    ----------
    hbo : array, shape (n_times,)
        Oxy-hemoglobin time series.
    hbr : array, shape (n_times,)
        Deoxy-hemoglobin time series.

    Returns
    -------
    hbo_corr : array, shape (n_times,)
        Corrected oxy-hemoglobin.
    hbr_corr : array, shape (n_times,)
        Corrected deoxy-hemoglobin.
    """
    # estimate scaling factor alpha = std(hbo)/std(hbr)
    alpha = np.std(hbo) / np.std(hbr)

    # compute the “true” hemodynamic component
    hemo = (hbo - alpha * hbr) / (1 + alpha)

    # reconstruct corrected signals
    hbo_corr = hemo
    hbr_corr = -hemo

    return hbo_corr, hbr_corr


# read the hdr file from preset line
def reading_events_hdr_file(filepath: str, start_line_number: int):
    """
       read the hd file to check
       the real stim events (that's) is not matching
       from the .nirs code taken from Matlab - MUST DO this for re-checking the event ids

       Parse a .hdr file to extract stimulus onsets and labels beyond a given start line.

       Parameters
       ----------
       filepath : str
          Path to the .hdr file.
       start_line_number : int
          Line number to begin reading events from.

       Returns
       -------
       events : ndarray, shape (n_events, 3)
          MNE-style event array [sample, 0, event_id].
       events_stim : list of str
          Corresponding event labels as 'Stim_<id>'.
    """

    dict_default = {
        'Stim_1': 0,
        'Stim_10': 0,
        'Stim_11': 0,
        'Stim_12': 0,
        'Stim_13': 0,
        'Stim_14': 0,
        'Stim_15': 0,
        'Stim_2': 0,
        'Stim_3': 0,
        'Stim_4': 0,
        'Stim_5': 0,
        'Stim_6': 0,
        'Stim_7': 0,
        'Stim_8': 0,
        'Stim_9': 0}
    events = []
    events_stim = []

    with open(filepath, 'r') as file:
        for current_line_num, line in enumerate(file, 1):
            if current_line_num >= start_line_number:
                if '#' in line.strip() and len(line.strip()) <= 2:
                    break

                print(f"Line {current_line_num}: {line.strip()}")
                value_stimuli = int(line.strip().split('\t')[1])
                latency_value = int(line.strip().split('\t')[2])
                if value_stimuli != 8:
                    events.append([latency_value, 0, value_stimuli])
                    events_stim.append("Stim_" + str(value_stimuli))

    logger.info(f"Processed the event calculation for {filepath}!!")

    return np.array(events), events_stim

# auxiliary plotting function


def plotting_box_plot_medians(
        data1: Any,
        data2: Any,
        data_all: Any,
        suffix: str):
    """
       Plotting function for getting the boxplot
       after median split

       Create and save a median-split boxplot comparing overall data, positives, and negatives.

       Parameters
       ----------
       data1 : array
         Values for "positive" group.
       data2 : array
         Values for "negative" group.
       data_all : array
         Combined values for reference.
       suffix : str
         Filename suffix for saving the plot.
    """

    plt.close("all")

    df = pd.DataFrame({
        'value': np.concatenate([data_all, data1, data2]),
        'group': ['HbO'] * len(data_all) + ['HbO positive'] * len(data1) + ['HbO negative'] * len(data2)
    })

    plt.figure(figsize=(10, 10))
    sns.set(style="whitegrid")

    # Boxplot
    ax = sns.boxplot(x='group', y='value', data=df,
                     width=0.6,
                     showfliers=False,
                     boxprops=dict(alpha=0.7),
                     medianprops=dict(color='firebrick', linewidth=2))

    # Add scatter of points
    sns.stripplot(x='group', y='value', data=df,
                  size=4, jitter=0.2, alpha=0.6, color='black', ax=ax)

    # Compute medians directly from the DataFrame
    meds = df.groupby('group')['value'].median()

    labels = [t.get_text() for t in ax.get_xticklabels()]
    positions = ax.get_xticks()
    x_positions = dict(zip(labels, positions))

    # First‐group median
    first_label = labels[0]     # change index if you want a different “first”
    x0 = x_positions[first_label]
    y0 = meds[first_label]

    for label in labels[1:]:
        xi = x_positions[label]
        yi = meds[label]
        ax.plot([x0, xi], [y0, yi],
                linestyle='--', marker='o', linewidth=2,
                color='navy')

    for label in ax.get_xticklabels():
        label.set_fontsize(15)

    for label in ax.get_yticklabels():
        label.set_fontsize(15)

    ax.set_xlabel('')
    ax.set_ylabel('uM.s', fontsize=15)
    ax.set_title('Median Split', fontsize=16)
    plt.tight_layout()
    plt.savefig("../figs_folder_auditory/median_split_" + suffix + '.jpg')


def plotting_average_response(data, title: str, suffix: str):
    """
       This function plots the average response across subjects
       and trials this will be done only for plotting.

       Plot and save the grand-average fNIRS response across epochs.

       This function:
        1. Computes the average time course across trials and channels.
        2. Calculates standard error across epochs and overlays shaded confidence bands.
        3. Formats axes, adds grids, and titles.
        4. Saves the figure under '../figs_folder/' with the given suffix.

       Parameters
       ----------
       data : mne.Epochs
          Epoched fNIRS data object containing epochs for one condition.
       title : str
          Figure title describing modality and condition.
       suffix : str
          Filename suffix for saving the plot image.

      Returns
      -------
      None
    """
    # Loop through all axes and modify appearance
    fig = data.average().plot(show=False)

    bad_channels = data.average().info["bads"]
    channel_names = data.average().ch_names

    # checking the intermediate channels
    indexes_to_use_hbo = [i for i, s in enumerate(
        channel_names) if s not in bad_channels and "hbo" in s]
    indexes_to_use_hbr = [i for i, s in enumerate(
        channel_names) if s not in bad_channels and "hbr" in s]

    n_epochs, n_channels, n_times = data._data.shape
    # shape (n_channels, n_times)
    sd = np.std(data._data * 1e6, axis=0)
    sem = sd / np.sqrt(n_epochs)                # get the standard error

    data_plotted = data.average()._data * 1e6
    time_plotted = data.average().times

    data_plotted_oxy = data_plotted[indexes_to_use_hbo, :]
    data_plotted_deoxy = data_plotted[indexes_to_use_hbr, :]
    count_axes = 0

    for ax in fig.axes:
        ax.grid(True, linewidth=0.5)  # Add grid

        if count_axes <= 1:
            if count_axes == 1:
                ax.set_xlabel("Time [s]", fontsize=17)
            ax.set_ylabel("uM", fontsize=17)
            ax.tick_params(axis='both', labelsize=15)
            ax.title.set_fontsize(16)

        for idx, line in enumerate(ax.lines):
            color = line.get_color()
            line.set_linewidth(3.0)  # Thicken lines

            # Draw the filled area under/over the curve
            if count_axes <= 1:
                if count_axes == 0:
                    ax.fill_between(
                        time_plotted,
                        data_plotted_oxy[idx, :] - sem[idx, :],
                        data_plotted_oxy[idx, :] + sem[idx, :],
                        color=color,
                        alpha=0.2
                    )
                else:
                    ax.fill_between(
                        time_plotted,
                        data_plotted_deoxy[idx, :] - sem[idx, :],
                        data_plotted_deoxy[idx, :] + sem[idx, :],
                        color=color,
                        alpha=0.2
                    )

        count_axes = count_axes + 1

    # Manually resize figure
    fig.set_size_inches(15, 8)  # width x height in inches
    fig.suptitle(title, fontsize=17)
    fig.savefig('../figs_folder_auditory/' + "average_" + suffix)

    logger.info(
        f"Plotting average of all signals for modality {title} and {suffix}..")

# get the HRF responses per subject


def calculate_subject_matrix_effects(
        data,
        stim_dur: float,
        drift_order: int,
        suffix: str,
        subject: str,
        stimuli: list,
        channel_index: int):
    """
       This function calculate here HRF estimates
       using a glover GLM and glover estimation

       Estimate subject-level HRF using GLM with Glover basis functions.

       Parameters
       ----------
       epochs : mne.Epochs
        Epoched fNIRS data.
       stim_dur : float
        Stimulus duration (s).
       drift_order : int
        Polynomial drift order.
       stimuli : list of str
        Event names to model.
       channel_index : int
        Index of the HbO channel for HRF extraction.

        Returns
        -------
        HRF_hbo, HRF_hbr : arrays
             Estimated average HRF for oxy and deoxy signals.
    """
    sfreq = data.info["sfreq"]
    frame_times = data.times
    len(frame_times)
    channels = data.ch_names

    hrf_B = glover_hrf(t_r=1, oversampling=sfreq, time_length=16)
    hrf_derivative = glover_time_derivative(
        t_r=1, oversampling=sfreq, time_length=16)
    hrf_derivative_dispersion = glover_dispersion_derivative(
        t_r=1, oversampling=sfreq, time_length=16)

    # uncomment this if you want to apply a FIR design matrix here
    # n_bins = 20 #int(16 * data.info["sfreq"])
    # design_matrix = make_first_level_design_matrix(
    #     raw_haemo,
    #     hrf_model='fir',
    #     fir_delays=list(range(n_bins)),
    #     drift_order=drift_order,
    #     stim_dur=stim_dur,
    #     drift_model="polynomial"
    # )

    # calculate the canonical glover response with derivative and dispersion,
    # then the HRF can be fully modelled using the glover response
    design_matrix = make_first_level_design_matrix(
        data,
        stim_dur=stim_dur,
        drift_model="polynomial",
        hrf_model="glover + derivative + dispersion",
        drift_order=drift_order)

    glm_results = run_glm(data, design_matrix)
    df_glm_results = glm_results.to_dataframe()
    rows_hbo_parameters = df_glm_results.loc[df_glm_results['ch_name']
                                             == channels[channel_index]]
    rows_hbr_parameters = df_glm_results.loc[df_glm_results['ch_name']
                                             == channels[channel_index + 44]]

    # do a for loop across the stimuli index in the matrix
    HRF_hbo = np.zeros((len(hrf_B)))
    HRF_hbr = np.zeros((len(hrf_B)))

    count_stim = 0
    for index_stim in range(0, len(stimuli)):
        theta_B = rows_hbo_parameters.loc[rows_hbo_parameters['Condition']
                                          == stimuli[index_stim]].iloc[0, 6]
        theta_derivative = rows_hbo_parameters.loc[rows_hbo_parameters['Condition']
                                                   == stimuli[index_stim] + '_derivative'].iloc[0, 6]
        theta_dispersion = rows_hbo_parameters.loc[rows_hbo_parameters['Condition']
                                                   == stimuli[index_stim] + '_dispersion'].iloc[0, 6]
        HRF_hbo = theta_B * hrf_B + theta_derivative * hrf_derivative + \
            theta_dispersion * hrf_derivative_dispersion + HRF_hbo

        theta_B_r = rows_hbo_parameters.loc[rows_hbo_parameters['Condition']
                                            == stimuli[index_stim]].iloc[0, 6]
        theta_derivative_r = rows_hbo_parameters.loc[rows_hbo_parameters['Condition']
                                                     == stimuli[index_stim] + '_derivative'].iloc[0, 6]
        theta_dispersion_r = rows_hbo_parameters.loc[rows_hbo_parameters['Condition']
                                                     == stimuli[index_stim] + '_dispersion'].iloc[0, 6]
        HRF_hbr = theta_B_r * hrf_B + theta_derivative_r * hrf_derivative + \
            theta_dispersion_r * hrf_derivative_dispersion + HRF_hbr
        count_stim = count_stim + 1

    HRF_hbo = HRF_hbo / count_stim
    HRF_hbr = HRF_hbr / count_stim

    # return the estimated HRF responses
    return HRF_hbo, HRF_hbr

# get here the auxiliary functions


def plotting_time_ch_fNIRs(
        raw_od,
        unit: str,
        title: str,
        suffix_arch: str,
        color_input=None):
    """
      doing the plotting of the fnirs obj
      around the timeseries

    Generate time-series plots for each fNIRS channel's optical density or hemoglobin signal.

    Process:
    1. Create subplots for paired HbO and HbR channel timeseries.
    2. Apply consistent styling: linewidths, legends, labels, and grids.
    3. Optionally use provided colors or random palette.
    4. Save the multi-panel figure under '../figs_folder/'.

    Parameters
    ----------
    raw_od : mne.io.Raw
        Raw optical density or haemoglobin data with paired channels.
    unit : str
        Y-axis label unit (e.g., 'AU' or 'uM').
    title : str
        Figure title.
    suffix_arch : str
        Filename suffix for saving the plot image.
    color_input : list of tuple, optional
        Predefined list of RGB tuples for each channel pair.

    Returns
    -------
    color_list : list of tuple
        List of RGB tuples used for plotting each channel pair.

    # Manually resize figure
     fig.set_size_inches(15, 8)  # width x height in inches
     fig.suptitle(title)
    """

    # get the time vector here
    time = np.linspace(
        0,
        (1 / raw_od.info["sfreq"]) * raw_od._data.shape[1],
        raw_od._data.shape[1])

    # plot the original signal here

    fig, axes = plt.subplots(
        nrows=14, ncols=1, figsize=(10, 8))

    count_axes = 0

    color_list = []

    for index_channels in range(0, 14, 1):
        if color_input is None:
            color = colors_tuple_list[count_axes]  # get_random_color()
            color_list.append(color)
        else:
            if count_axes <= len(color_input) - 1:
                color = color_input[count_axes]
            else:
                color = colors_tuple_list[count_axes]  # get_random_color()

        axes[count_axes].plot(time,
                              raw_od._data[index_channels,
                                           :] * 1e6,
                              linewidth=3,
                              color=color,
                              label="HbO")
        axes[count_axes].legend()
        # axes[count_axes].set_xlabel("Time [s]")
        axes[count_axes].set_ylabel(unit)
        axes[count_axes].grid(True)
        # axes[count_axes].twinx()

        color_new = np.subtract(np.array(color), [0.2, 0.2, 0.2])
        # Check for underflow and handle as needed (e.g., setting negative
        # values to 0)
        color_new[color_new < 0] = 0

        axes[count_axes].plot(time,
                              raw_od._data[index_channels + 44,
                                           :] * 1e6,
                              linewidth=3,
                              color=color_new,
                              linestyle="dotted",
                              label="HbR")
        axes[count_axes].legend()
        # axes[count_axes].set_xlabel("Time [s]")
        axes[count_axes].set_ylabel(unit)
        axes[count_axes].set_title(
            raw_od.ch_names[count_axes].split(' ')[0], fontsize=10)
        axes[count_axes].grid(True)

        count_axes = count_axes + 1

    axes[count_axes - 1].set_xlabel("Time [s]")
    fig.tight_layout()
    fig.suptitle(title)
    fig.savefig(
        '../figs_folder_auditory/' +
        "time_series_" +
        suffix_arch +
        ".jpg")

    return color_list
    # plot here the 400s in the mne-nirs plot object
    # raw_od.plot(duration=400, show_scrollbars=False)


sel_plot = int(sys.argv[1])
sel_method = int(sys.argv[2])
sel_load = int(sys.argv[4])

# define the stim types depending what Kurt Steinmetzger report in the
# email dividing between the five stimuli type (categories) reported in
# the paper
stim_types = [
    ["Stim_1"], ["Stim_2"], ["Stim_5"], [
        "Stim_3", "Stim_4"], [
            "Stim_6", "Stim_7"]]
stim_types_string = [
    "No_Pitch",
    "Strong_Pitch_Dynamic",
    "Strong_Pitch_Static",
    "Weak_Pitch_Dynamic",
    "Weak_Pitch_Static"]

data_folder = "../fNIRS data/"
data_folder_eeg = "../EEG data/"

# define here the global variables..
SUBJECTS = []
DATA_BLOCK = []
DATA_STIM = []
bad_CH_NAMES = []
auc_values = np.zeros((20, 5))
trials_per_subject = np.zeros((20, 5))

HRF_responses = []

temp_raw_od = []

# do the for loop across all the subjects and stimili type
for index_stimuli in range(0, len(stim_types)):
    # get the subject counter to 0
    count_subject = 0

    data_BLOCK = []
    data_STIM = []
    subjects = []
    bad_ch_names = []
    hrf_responses = []

    # read the data from the converted snirf file here
    for dirpath, dirnames, filenames in os.walk(data_folder):
        for dirname in dirnames:
            # read the fnirs files associated to the Matlab created snirf files
            # (derivatives are already created)
            if '-' in dirname and not ('_' in dirname) and sel_load == 0:
                # Define filter parameters
                l_freq = 0.01  # 0.1 low-pass filter
                h_freq = 0.5  # High-pass filter cutoff at 0.2 Hz
                h_trans_bandwidth = 0.5  # Transition bandwidth

                if os.path.isdir(
                    data_folder +
                    dirname +
                    '/' +
                    dirname +
                        "_001/"):
                    data_folder_sub = data_folder + dirname + '/' + dirname + "_001/"
                    data_folder_sub_eeg = data_folder_eeg + dirname + '/' + dirname + "_001/"
                    raw = read_raw_snirf(
                        data_folder_sub +
                        "NIRS-" +
                        dirname +
                        "_001.snirf",
                        preload=True)
                    events, event_dict = reading_events_hdr_file(
                        filepath=data_folder_sub + "NIRS-" + dirname + "_001.hdr", start_line_number=53)
                elif os.path.isdir(data_folder + dirname + '/' + dirname + "_002/"):
                    data_folder_sub = data_folder + dirname + '/' + dirname + "_002/"
                    data_folder_sub_eeg = data_folder_eeg + dirname + '/' + dirname + "_002/"
                    raw = read_raw_snirf(
                        data_folder_sub +
                        "NIRS-" +
                        dirname +
                        "_002.snirf",
                        preload=True)
                    events, event_dict = reading_events_hdr_file(
                        filepath=data_folder_sub + "NIRS-" + dirname + "_002.hdr", start_line_number=53)
                elif os.path.isdir(data_folder + dirname + '/' + dirname + "_003/"):
                    data_folder_sub = data_folder + dirname + '/' + dirname + "_003/"
                    data_folder_sub_eeg = data_folder_eeg + dirname + '/' + dirname + "_003/"
                    raw = read_raw_snirf(
                        data_folder_sub +
                        "NIRS-" +
                        dirname +
                        "_003.snirf",
                        preload=True)
                    events, event_dict = reading_events_hdr_file(
                        filepath=data_folder_sub + "NIRS-" + dirname + "_003.hdr", start_line_number=53)
                elif os.path.isdir(data_folder + dirname + '/' + dirname + "_004/"):
                    data_folder_sub = data_folder + dirname + '/' + dirname + "_004/"
                    data_folder_sub_eeg = data_folder_eeg + dirname + '/' + dirname + "_004/"
                    raw = read_raw_snirf(
                        data_folder_sub +
                        "NIRS-" +
                        dirname +
                        "_004.snirf",
                        preload=True)
                    events, event_dict = reading_events_hdr_file(
                        filepath=data_folder_sub + "NIRS-" + dirname + "_004.hdr", start_line_number=53)

                # get the EEG information here for getting the right
                # information from the EEG
                files_eeg = [
                    files for files in listdir(data_folder_sub_eeg) if isfile(
                        join(
                            data_folder_sub_eeg,
                            files))]

                # read the real events here
                for index_file in range(0, len(files_eeg)):
                    data_EEG = load_matlab_file(
                        path=data_folder_sub_eeg + files_eeg[index_file])

                trials_real = np.squeeze(data_EEG["data"]["trialinfo"][0][0])

                # get the events here
                events_stim = []
                for index_real_trials in range(0, len(trials_real)):
                    events_stim.append("Stim_" +
                                       str(int(trials_real[index_real_trials])))

                # getting raw optical density from the raw data here
                raw_od = mne.preprocessing.nirs.optical_density(raw)

                # plotting the raw_od here before doing the motion correction
                if index_stimuli == 0:
                    color_list = plotting_time_ch_fNIRs(
                        raw_od=raw_od,
                        unit="AU",
                        title=f"uncorrected optical density {dirname}",
                        suffix_arch=dirname + "_raw")

                # %% scalp coupling index ## do this rejection provisionally without employing manual channel rejection
                sci = mne.preprocessing.nirs.scalp_coupling_index(
                    raw_od, l_freq=0.5)

                logger.info(
                    f"The scalp coupling across channels for {dirname} is {sci}")

                # %% remove bad channels based on the scalp coupling index
                raw_od.info['bads'] = list(
                    compress(raw_od.ch_names, sci < 0.75))

                # log here the removed channels
                logger.info(['Removed', raw_od.info['bads']])

                suffix_arch = "tddr"

                if sel_method == 0:
                    raw_od = mne.preprocessing.nirs.temporal_derivative_distribution_repair(
                        raw_od)

                # do here the kalman filter
                if sel_method == 1:
                    suffix_arch = "kalman"
                    if index_stimuli != 0:
                        raw_od = temp_raw_od[count_subject]
                    else:
                        for index_ch in range(0, raw_od._data.shape[0]):
                            # Optionally: learn better covariances via EM
                            kf = kf.em(raw_od._data[index_ch, :], n_iter=20)
                            # Run smoothing (or filtering)
                            smoothed_values, _ = kf.smooth(
                                raw_od._data[index_ch, :])
                            raw_od._data[index_ch, :] = np.squeeze(
                                smoothed_values)

                            logger.info(
                                f"applying Kalman filter for {
                                    raw_od.ch_names[index_ch]}")
                        temp_raw_od.append(raw_od)

                # %% convert to changes in oxy/deoxy
                raw_haemo = mne.preprocessing.nirs.beer_lambert_law(
                    raw_od, ppf=1.0)

                # applying cbs_i one here..
                if sel_method == 2:
                    suffix_arch = "cbs_i"
                    for index_ch in range(
                            0,
                            round(
                                raw_haemo._data.shape[0] /
                                2),
                            1):
                        raw_haemo._data[index_ch, :], raw_haemo._data[index_ch + (44 - len(raw_od.info['bads']) - 1), :] = cbs_i(
                            raw_haemo._data[index_ch, :], raw_haemo._data[index_ch + (44 - len(raw_od.info['bads']) - 1), :])

                # do the plotting of the first raw optical density
                if index_stimuli == 0:
                    plotting_time_ch_fNIRs(
                        raw_od=raw_haemo,
                        unit="uM",
                        title=f"signal after artifact removal before filter {dirname}",
                        suffix_arch=dirname + "_" + suffix_arch,
                        color_input=color_list)

                # Apply the filter after applying beer lambert law conversion
                raw_haemo = raw_haemo.copy().filter(
                    l_freq=l_freq,
                    h_freq=h_freq,
                    h_trans_bandwidth=h_trans_bandwidth,
                    verbose=True)

                # apply this for signal enhacement
                raw_haemo = mne_nirs.signal_enhancement.enhance_negative_correlation(
                    raw_haemo.copy())

                # set here the new event annotation
                if len(events[:, 0]) != len(events_stim):
                    events_stim = event_dict
                new_annots = mne.Annotations(
                    onset=events[:, 0] / raw.info["sfreq"], duration=events[:, 1], description=events_stim)

                raw.set_annotations(new_annots)
                raw_haemo.set_annotations(new_annots)

                # re-sample if necessary if sampling frequency is different
                # than 7.8Hz
                if raw_haemo.info["sfreq"] != 7.8:
                    raw_haemo.resample(7.8, npad='auto')

                # do the plotting of the first raw optical density
                if index_stimuli == 0:
                    plotting_time_ch_fNIRs(
                        raw_od=raw_haemo,
                        unit="uM",
                        title=f"signal after artifact removal after filter {dirname}",
                        suffix_arch=dirname +
                        "_filter_" +
                        suffix_arch,
                        color_input=color_list)

                # comment this for doing the real stimuli id for each subject -
                # This is not the right one from Matlab!
                events_new, event_dict_new = mne.events_from_annotations(raw)
                # breakpoint()

                # epoched data
                epochs_stimuli = mne.Epochs(
                    raw_haemo,
                    events_new,
                    event_id={
                        e: event_dict_new[e] for e in stim_types[index_stimuli]},
                    tmin=0,
                    tmax=32,
                    baseline=None,
                    decim=1,
                    detrend=1,
                    event_repeated="merge",
                    preload=True,
                    verbose=True)

                # truncate the epochs for training purposes
                if epochs_stimuli._data.shape[0] != 180:
                    epochs_stimuli = epochs_stimuli[:180]

                trials_per_subject[count_subject,
                                   index_stimuli] = epochs_stimuli._data.shape[0]

                # do here the median split for checking what subject will be defined
                # adding the AUC for channels 34 and 36
                auc_val_34 = np.trapz(np.mean(epochs_stimuli._data[:, 34, round(epochs_stimuli._data.shape[2] / 4):round(
                    epochs_stimuli._data.shape[2])], axis=0), np.linspace(8, 32, round(epochs_stimuli._data.shape[2] * 0.75)))
                auc_val_36 = np.trapz(np.mean(epochs_stimuli._data[:, 36, round(epochs_stimuli._data.shape[2] / 4):round(
                    epochs_stimuli._data.shape[2])], axis=0), np.linspace(8, 32, round(epochs_stimuli._data.shape[2] * 0.75)))
                auc_val = np.trapz(np.mean(epochs_stimuli._data[:, int(sys.argv[3]), round(epochs_stimuli._data.shape[2] / 4):round(
                    epochs_stimuli._data.shape[2] / 2)], axis=0), np.linspace(8, 16, round(epochs_stimuli._data.shape[2] / 4)))
                auc_values[count_subject, index_stimuli] = np.mean(
                    [auc_val_34, auc_val_36])
                # auc_values[count_subject, index_stimuli] = auc_val

                # epoched data
                epochs_stimuli_stim = mne.Epochs(
                    raw_haemo,
                    events_new,
                    event_id={
                        e: event_dict_new[e] for e in stim_types[index_stimuli]},
                    tmin=0,
                    tmax=1,
                    baseline=None,
                    decim=1,
                    detrend=1,
                    event_repeated="merge",
                    preload=True,
                    verbose=True)

                # truncate the epochs for training purposes
                if epochs_stimuli_stim._data.shape[0] != 180:
                    epochs_stimuli_stim = epochs_stimuli_stim[:180]

                logger.info(
                    f"Calculating the design matrix for {dirname} and stim_type {
                        stim_types_string[index_stimuli]}")
                # here we calculate HRF response
                if sel_plot == 1:
                    HRF_hbo, HRF_hbr = calculate_subject_matrix_effects(
                        data=raw_haemo, stim_dur=16.0, drift_order=5, suffix=dirname + "_" + suffix_arch, subject=dirname, stimuli=stim_types[index_stimuli], channel_index=int(sys.argv[3]))
                    hrf_responses.append([HRF_hbo, HRF_hbr])

                # get the common bads here between each subject
                if count_subject >= 1:
                    common_bads = list(set(data_block.info['bads']) | set(
                        epochs_stimuli.info['bads']))
                    data_block.info['bads'] = common_bads
                    data_stim.info['bads'] = common_bads
                    epochs_stimuli.info['bads'] = common_bads
                    epochs_stimuli_stim.info['bads'] = common_bads
                    data_block = concatenate_epochs(
                        [data_block, epochs_stimuli])
                    data_stim = concatenate_epochs(
                        [data_stim, epochs_stimuli_stim])
                else:
                    data_block = epochs_stimuli
                    data_stim = epochs_stimuli_stim

                bad_ch_names.append(data_block.info['bads'])
                data_BLOCK.append(data_block._data)
                data_STIM.append(data_stim._data)

                # epochs_stimuli.average().plot()
                logger.info(f"Reading fNIRs information from {dirname}")
                logger.info(
                    f"fNIRs information details from {dirname} is {
                        raw.info} and stim_type {
                        stim_types_string[index_stimuli]}")

            if '-' in dirname and not ('_' in dirname):
                count_subject = count_subject + 1
                subjects.append(dirname)

    # append the HRF responses
    if sel_plot == 1:
        HRF_responses.append(hrf_responses)
        # save the temporal HRF_responses here in case you need to save it
        with open('../temporal_hrf_responses_' + f'{int(sys.argv[3])}.pkl', 'wb') as file:
            pickle.dump([HRF_responses, hrf_responses], file)

    # append the data here before saving it in the npz
    if count_subject >= 20:
        if sel_load == 0:
            DATA_BLOCK.append(data_BLOCK[count_subject - 1])
            DATA_STIM.append(data_STIM[count_subject - 1])
            bad_CH_NAMES = bad_ch_names[count_subject - 1]
        SUBJECTS = subjects

    # plot the wrap all the subjects across the stimuli type, so did to bad
    # channels wrap up first
    if sel_load == 0:
        plotting_average_response(
            data=data_block,
            title=stim_types_string[index_stimuli].replace(
                "_",
                " "),
            suffix=suffix_arch +
            stim_types_string[index_stimuli].replace(
                "_",
                " "))


if sel_load == 0:
    # do the ten & ten subject split using the median split on HbO positive
    # and HbO negative
    auc_values_sub = np.mean(auc_values, axis=1)
    sorted_with_indices = sorted(
        enumerate(auc_values_sub),
        key=lambda x: x[1],
        reverse=True)

    # Extract the sorted values and their original indices
    sorted_values = [value for index, value in sorted_with_indices]
    indices_subject = [index for index, value in sorted_with_indices]

    indices_hbo_positive = indices_subject[0:10]
    indices_hbo_negative = indices_subject[10:]

    # multiply the value for give this in uM.s
    auc_values_sub = auc_values_sub * 1e6

    # plotting here the median split..
    plotting_box_plot_medians(
        data1=auc_values_sub[indices_hbo_positive],
        data2=auc_values_sub[indices_hbo_negative],
        data_all=auc_values_sub,
        suffix=suffix_arch)

    # moving around the range of the subjects around...
    data_hbo_positive = np.zeros((np.array(DATA_BLOCK[0]).shape[2]))
    data_hbo_negative = np.zeros((np.array(DATA_BLOCK[0]).shape[2]))
    data_hbo_positive_sd = np.zeros((np.array(DATA_BLOCK[0]).shape[2]))
    data_hbo_negative_sd = np.zeros((np.array(DATA_BLOCK[0]).shape[2]))

    # get here the hbo positive/negative values from the trials
    for index_stimuli in range(0, 5):
        for index_sub in range(0, len(indices_hbo_positive)):
            data_hbo_positive = data_hbo_positive + np.mean(np.array(DATA_BLOCK[index_stimuli])[int(np.sum(trials_per_subject[0:indices_hbo_positive[index_sub], index_stimuli])):int(np.sum(
                trials_per_subject[0:indices_hbo_positive[index_sub], index_stimuli]) + trials_per_subject[indices_hbo_positive[index_sub], index_stimuli]), int(sys.argv[3]), :], axis=0) * 1e6
            data_hbo_negative = data_hbo_negative + np.mean(np.array(DATA_BLOCK[index_stimuli])[int(np.sum(trials_per_subject[0:indices_hbo_negative[index_sub], index_stimuli])):int(np.sum(
                trials_per_subject[0:indices_hbo_negative[index_sub], index_stimuli]) + trials_per_subject[indices_hbo_negative[index_sub], index_stimuli]), int(sys.argv[3]), :], axis=0) * 1e6
            data_hbo_positive_sd = data_hbo_positive_sd + np.std(np.array(DATA_BLOCK[index_stimuli])[int(np.sum(trials_per_subject[0:indices_hbo_positive[index_sub], index_stimuli])):int(
                np.sum(trials_per_subject[0:indices_hbo_positive[index_sub], index_stimuli]) + trials_per_subject[indices_hbo_positive[index_sub], index_stimuli]), int(sys.argv[3]), :], axis=0) * 1e6
            data_hbo_negative_sd = data_hbo_negative_sd + np.std(np.array(DATA_BLOCK[index_stimuli])[int(np.sum(trials_per_subject[0:indices_hbo_negative[index_sub], index_stimuli])):int(
                np.sum(trials_per_subject[0:indices_hbo_negative[index_sub], index_stimuli]) + trials_per_subject[indices_hbo_negative[index_sub], index_stimuli]), int(sys.argv[3]), :], axis=0) * 1e6

    data_hbo_positive = data_hbo_positive / 50
    data_hbo_negative = data_hbo_negative / 50
    sem_hbo_positive = data_hbo_positive_sd / (50 * np.sqrt(50))
    sem_hbo_negative = data_hbo_negative_sd / (50 * np.sqrt(50))

    # after 32 seconds of usage across the entire block
    time_vals = np.linspace(0, 32, np.array(DATA_BLOCK[0]).shape[2])

    plt.close("all")

    # set the size of the figure here
    fig, axes = plt.subplots(
        nrows=1, ncols=1, figsize=(10, 10))

    axes.plot(
        time_vals,
        data_hbo_positive,
        linewidth=3,
        color="red",
        label="HbO Positive")
    axes.plot(
        time_vals,
        data_hbo_negative,
        linewidth=3,
        color="blue",
        label="HbO Negative")
    axes.fill_between(
        time_vals,
        data_hbo_positive -
        sem_hbo_positive /
        2,
        data_hbo_positive +
        sem_hbo_positive /
        2,
        color="red",
        alpha=0.2)
    axes.fill_between(
        time_vals,
        data_hbo_negative -
        sem_hbo_negative /
        2,
        data_hbo_negative +
        sem_hbo_negative /
        2,
        color="blue",
        alpha=0.2)

    axes.set_xlabel("Time [s]", fontsize=15)
    axes.set_ylabel("uM", fontsize=15)

    # Modify tick label font sizes individually
    for label in axes.get_xticklabels():
        label.set_fontsize(12)

    for label in axes.get_yticklabels():
        label.set_fontsize(12)

    axes.set_title(f"Channel {int(sys.argv[3])}", fontsize=17)
    axes.grid(True)
    axes.legend()

    fig.savefig('../figs_folder_auditory/' +
                "plot_HbO_positive_negative_" +
                suffix_arch +
                f"{int(sys.argv[3])}.jpg")

    plt.close("all")

    # plotting the HRF values given the model itself
    if sel_plot == 1:
        HRF_responses = np.array(HRF_responses)

        hrf_positive = np.mean(
            np.mean(HRF_responses[:, indices_hbo_positive, 0, :], axis=0), axis=0) * 10e6
        hrf_negative = np.mean(
            np.mean(HRF_responses[:, indices_hbo_negative, 0, :], axis=0), axis=0) * 10e6

        hrf_positive_sd = np.std(
            np.std(HRF_responses[:, indices_hbo_positive, 0, :], axis=0), axis=0) * 10e6
        hrf_negative_sd = np.std(
            np.std(HRF_responses[:, indices_hbo_negative, 0, :], axis=0), axis=0) * 10e6

        sem_hrf_positive = hrf_positive_sd / np.sqrt(50)
        sem_hrf_negative = hrf_negative_sd / np.sqrt(50)

        # get here the time vals for this
        time_vals = np.linspace(0, 32, HRF_responses.shape[3])

        # plotting the modelled response here..
        fig, axes = plt.subplots(
            nrows=1, ncols=1, figsize=(10, 10))

        axes.plot(
            time_vals,
            hrf_positive,
            linewidth=3,
            color="red",
            label="HbO Positive")
        axes.plot(
            time_vals,
            hrf_negative,
            linewidth=3,
            color="blue",
            label="HbO Negative")
        axes.fill_between(
            time_vals,
            hrf_positive -
            sem_hrf_positive,
            hrf_positive +
            sem_hrf_positive,
            color="red",
            alpha=0.2)
        axes.fill_between(
            time_vals,
            hrf_negative -
            sem_hrf_negative,
            hrf_negative +
            sem_hrf_negative,
            color="blue",
            alpha=0.2)

        axes.set_title(f"HRF modelled Channel {int(sys.argv[3])}", fontsize=17)

        axes.set_xlabel("Time [s]", fontsize=15)
        axes.set_ylabel("uM", fontsize=15)

        # Modify tick label font sizes individually
        for label in axes.get_xticklabels():
            label.set_fontsize(12)

        for label in axes.get_yticklabels():
            label.set_fontsize(12)

        axes.grid(True)
        axes.legend()

        fig.savefig('../figs_folder_auditory/' +
                    "plot_HRF_positive_negative_" +
                    suffix_arch +
                    f"{int(sys.argv[3])}.jpg")

        logger.info("Saving data files...")

# save the data here using npz format # John can you tell me if the file
# structure is the one we need?
if sel_load == 0:
    np.savez_compressed('../processed_data_' + suffix_arch + '_stim_wise.npz',
                        data_no_pitch=np.array(DATA_BLOCK[0]),
                        data_strong_pitch_dynamic=np.array(DATA_BLOCK[1]),
                        data_strong_pitch_static=np.array(DATA_BLOCK[2]),
                        data_weak_pitch_dynamic=np.array(DATA_BLOCK[3]),
                        data_weak_pitch_static=np.array(DATA_BLOCK[4]),
                        data_no_pitch_stim=np.array(DATA_STIM[0]),
                        data_strong_pitch_dynamic_stim=np.array(DATA_STIM[1]),
                        data_strong_pitch_static_stim=np.array(DATA_STIM[2]),
                        data_weak_pitch_dynamic_stim=np.array(DATA_STIM[3]),
                        data_weak_pitch_static_stim=np.array(DATA_STIM[4]),
                        subjects=np.array(SUBJECTS),
                        bad_channels=np.array(bad_CH_NAMES),
                        stim_type=np.array(stim_types_string),
                        trials_per_subject=trials_per_subject,
                        indices_hbo_positive=indices_hbo_positive,
                        indices_hbo_negative=indices_hbo_negative,
                        auc_values_sub=auc_values_sub,
                        sorted_with_indices=sorted_with_indices)
else:
    # do here the kalman filter
    if sel_method == 1:
        suffix_arch = "kalman"
    elif sel_method == 2:
        suffix_arch = "cbs_i"
    else:
        suffix_arch = "tddr"
    filename = '../DATA/fNIRs/processed_data_' + suffix_arch + '_stim_wise.npz'
    data_load = np.load(filename)
    data_dict_load = dict(data_load)
    data_dict_load['subjects'] = SUBJECTS
    # overwriting
    np.savez(filename, **data_dict_load)  # Overwrites the original file

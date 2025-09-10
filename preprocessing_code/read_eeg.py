"""
    Module: read_eeg.py
    An example of reading EEG signals using MNE package - for the Steinmetzger_et_al datasets.

    This script processes raw EEG recordings in MATLAB format for auditory pitch stimuli,
    applying filtering, resampling, artifact correction (notch, ASR, ICA), and saving
    both raw and preprocessed plots per trial. It aggregates data by condition and persists
    the cleaned dataset for later analysis.

    Usage:
    python read_eeg.py <flag>. This flage is related to a previous file saved and if the code.. will start again
    from the previous saved file. Like a saved checkpoint.

    Arguments:
        flag (int): If 1 and a preprocessed subject file exists, loads from pickle; otherwise,
                performs preprocessing across all subject directories in '../EEG data/'.

    Outputs:
    - JPEG plots of raw and preprocessed EEG for each trial under '../figs_eeg_auditory/'.
    - Pickle files '../EEG_pre_processed_subject.pkl' and '../EEG_pre_processed.pkl' containing
      lists of condition-specific EEG arrays and subject identifiers.

"""

import asrpy
import mne
import sys
import pickle
import os
import warnings
import pandas as pd
import seaborn as sns
import numpy as np
import h5py
from scipy import io
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.stats.multitest as smm
from matplotlib import pyplot as plt
from mne_bids import BIDSPath, read_raw_bids
from loguru import logger
from os import listdir
from os.path import isfile, join
from scipy.stats import gaussian_kde
from statsmodels.stats.anova import AnovaRM
import statsmodels.formula.api as smf


# get here the auxiliary functions
def load_matlab_file(path):
    """
       Load a .mat file, v7 or earlier via scipy.io.loadmat,
       or v7.3+ via h5py.

       Parameters:
        path (str): Path to the .mat file.

       Returns:
        Dict[str, Any]: Mapping from variable names to numpy arrays.

       Raises:
        FileNotFoundError: If the file does not exist at the given path.
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


warnings.filterwarnings("ignore")

data_folder = "../EEG data/"

"""
    Main execution block:
    - If invoked with flag=1 and a per-subject pickle exists, load data from disk.
    - Otherwise, traverse '../EEG data/' folders, load and preprocess each trial:
        * Channel selection and montage setup
        * Resampling to 512 Hz and bandpass/notch filtering
        * Mean-centering
        * ASR artifact correction
        * ICA to remove ocular/muscle artifacts
        * Saving raw and preprocessed trial plots
     - Aggregate cleaned data by condition and save to pickle files.
 """

# here we start the main body of the code and execution
if int(sys.argv[1]) == 1 and os.path.exists(
        '../EEG_pre_processed_subject.pkl'):

    with open('../EEG_pre_processed.pkl', 'rb') as file_all:
        DATA = pickle.load(file_all)
        EEG_no_pitch = DATA[0]
        EEG_strong_pitch_dynamic = DATA[1]
        EEG_strong_pitch_static = DATA[2]
        EEG_weak_pitch_dynamic = DATA[3]
        EEG_weak_pitch_static = DATA[4]
        subject = DATA[5]

    logger.info("Reading previous file here!!..")
else:
    # lists with the append data after the pre-processing is executed..
    EEG_no_pitch = []
    EEG_strong_pitch_dynamic = []
    EEG_strong_pitch_static = []
    EEG_weak_pitch_dynamic = []
    EEG_weak_pitch_static = []
    subject = []

count_subject = 0
# read the EEG files here based on directories path
# run here the subjects moving across the folder..
for dirpath, dirnames, filenames in os.walk(data_folder):
    for dirname in dirnames:
        logger.info(f"processing {dirname}")
        # read the fnirs files associated to the Matlab created snirf files
        # (derivatives are already created)
        # and dirname == "2019-05-04": #and dirname == "2019-04-30": # and
        # dirname == "2019-01-27":
        if '-' in dirname and not (
                '_' in dirname) and count_subject >= len(subject):

            EEG_no_pitch_sub = []
            EEG_strong_pitch_dynamic_sub = []
            EEG_strong_pitch_static_sub = []
            EEG_weak_pitch_dynamic_sub = []
            EEG_weak_pitch_static_sub = []

            if os.path.isdir(data_folder + dirname + '/' + dirname + "_001/"):
                data_folder_sub = data_folder + dirname + '/' + dirname + "_001/"
            elif os.path.isdir(data_folder + dirname + '/' + dirname + "_002/"):
                data_folder_sub = data_folder + dirname + '/' + dirname + "_002/"
            elif os.path.isdir(data_folder + dirname + '/' + dirname + "_003/"):
                data_folder_sub = data_folder + dirname + '/' + dirname + "_003/"
            elif os.path.isdir(data_folder + dirname + '/' + dirname + "_004/"):
                data_folder_sub = data_folder + dirname + '/' + dirname + "_004/"

            files_eeg = [
                files for files in listdir(data_folder_sub) if isfile(
                    join(
                        data_folder_sub,
                        files))]

            for index_file in range(0, len(files_eeg)):

                data_EEG = load_matlab_file(
                    path=data_folder_sub + files_eeg[index_file])

                # get the information related to each
                num_trials = data_EEG["data"]["trialinfo"][0][0].shape[0]
                channels = data_EEG["data"]["label"][0][0]

                CHANNELS = []
                for index_chan in range(0, channels.shape[0]):
                    CHANNELS.append(channels[index_chan][0][0])

                # remove the non-existent channels
                removed_indices = [
                    CHANNELS.index("HEOG1"),
                    CHANNELS.index("HEOG2"),
                    CHANNELS.index("VEOG1"),
                    CHANNELS.index("VEOG2")]
                for idx_rm in sorted(removed_indices, reverse=True):
                    CHANNELS.pop(idx_rm)

                CHANNELS[CHANNELS.index("PO7'")] = "PO7"
                CHANNELS[CHANNELS.index("PO8'")] = "PO8"

                # get here the montage for each mne raw element
                montage = mne.channels.make_standard_montage('standard_1005')

                all_positions = montage.get_positions()['ch_pos']

                selected_positions = [all_positions[ch]
                                      for ch in CHANNELS if ch in all_positions]

                # get the montage for the new type of mne array
                new_montage = mne.channels.make_dig_montage(
                    ch_pos=dict(zip(CHANNELS, selected_positions)))

                info = mne.create_info(
                    ch_names=CHANNELS,
                    sfreq=256,  # sampling frequency is 256 by default from the .mat file
                    ch_types='eeg'
                )

                info.set_montage(new_montage)

                # read here the data related to a particula file
                DATA_EEG = []
                trial_event = []
                # do the pre-processing for each trial in the project
                for index_data in range(0, num_trials):
                    # run here the EEG data pre-processing
                    EEG_data = data_EEG["data"]["trial"][0][0][0][index_data][:, 77:] * 1e-6
                    trial_event.append(
                        int(data_EEG["data"]["trialinfo"][0][0][index_data][0]))
                    trial_value = int(
                        data_EEG["data"]["trialinfo"][0][0][index_data][0])
                    DATA_EEG.append(
                        data_EEG["data"]["trial"][0][0][0][index_data][:, 77:] * 1e-6)

                    # take here the mask for removing the rows corresponding to
                    # the removed channels
                    mask = np.ones(EEG_data.shape[0], dtype=bool)
                    mask[removed_indices] = False   # drop rows 1 and 3
                    EEG_data = EEG_data[mask, :]

                    EEG_raw_val = mne.io.RawArray(EEG_data, info)
                    # first filter the signal between 0.1 and 15Hz based on
                    # paper
                    EEG_raw_val.resample(512, npad="auto")
                    EEG_raw_val.filter(0.1, 15, fir_design='firwin')
                    EEG_raw_val.notch_filter(freqs=[60], fir_design='firwin')
                    EEG_raw_val.notch_filter(freqs=[50], fir_design='firwin')

                    # one mean per channel, shape (n_channels,)
                    means = EEG_raw_val._data.mean(axis=1)

                    # subtract each channel’s mean
                    EEG_raw_val._data -= means[:, np.newaxis]

                    fig_raw = EEG_raw_val.plot(
                        scalings="auto",
                        show_scrollbars=False,
                        show_scalebars=True,
                        show=False)
                    fig_raw.savefig(
                        f"../figs_eeg_auditory/raw_{index_data}_{trial_value}.jpg")
                    plt.close("all")

                    # process ICA here for each trial
                    ica = mne.preprocessing.ICA(
                        n_components=59,
                        random_state=42,
                        method="fastica",
                        max_iter=200000,
                        fit_params={
                            'tol': 5e-4})

                    EEG_raw_val._data[EEG_raw_val._data == 0.0] = 1e-15

                    # get a copy of the mne raw object
                    EEG_cleaned = EEG_raw_val.copy()

                    # apply ASR here and take into account that each trial ahs
                    # a length of 0.5s or 500ms so make the win_len and the
                    # win_overlap smaller
                    asr = asrpy.ASR(
                        sfreq=EEG_cleaned.info["sfreq"],
                        cutoff=25,
                        win_len=0.0122,
                        win_overlap=0.005)
                    ret_value = asr.fit(EEG_cleaned, return_clean_window=True)
                    if ret_value is None:
                        asr = asrpy.ASR(
                            sfreq=EEG_cleaned.info["sfreq"],
                            cutoff=25,
                            win_len=0.0122,
                            win_overlap=0.005,
                            max_dropout_fraction=0.05)
                        ret_value = asr.fit(
                            EEG_cleaned, return_clean_window=True)

                    EEG_cleaned = asr.transform(EEG_cleaned)

                    # Fit ICA to ASR-cleaned data
                    ica.fit(EEG_cleaned)

                    eog_inds, scores = ica.find_bads_eog(
                        EEG_cleaned, ch_name=["Fp1", "Fp2", "AFz"])
                    muscle_inds, scores = ica.find_bads_muscle(
                        EEG_cleaned, threshold=0.5, l_freq=20, h_freq=75)

                    ica.exclude = eog_inds + muscle_inds

                    logger.info(
                        f"excluded ICs are {
                            ica.exclude} for trial {index_data}")

                    # Apply ICA to remove artifacts
                    EEG_cleaned = ica.apply(EEG_cleaned)
                    EEG_cleaned._data[EEG_cleaned._data >= 200e-6] = 1e-15
                    EEG_cleaned._data[EEG_cleaned._data <= -200e-6] = -1e-15

                    fig_preprocessed = EEG_cleaned.plot(
                        scalings="auto", show_scrollbars=False, show_scalebars=True, show=False)
                    fig_preprocessed.savefig(
                        f"../figs_eeg_auditory/preprocessed_{index_data}_{trial_value}.jpg")
                    plt.close("all")

                    if trial_value == 1:
                        EEG_no_pitch_sub.append(EEG_cleaned._data)
                    elif trial_value == 2:
                        EEG_strong_pitch_dynamic_sub.append(EEG_cleaned._data)
                    elif trial_value == 5:
                        EEG_strong_pitch_static_sub.append(EEG_cleaned._data)
                    elif trial_value == 3 or trial_value == 4:
                        EEG_weak_pitch_dynamic_sub.append(EEG_cleaned._data)
                    elif trial_value == 6 or trial_value == 7:
                        EEG_weak_pitch_static_sub.append(EEG_cleaned._data)

                    logger.info(
                        f"Trial processed {index_data} with trial_label {trial_value}")

                    if index_data % 1000 == 0:
                        with open('../EEG_pre_processed_subject.pkl', 'wb') as file:
                            pickle.dump([EEG_no_pitch_sub,
                                         EEG_strong_pitch_dynamic_sub,
                                         EEG_strong_pitch_static_sub,
                                         EEG_weak_pitch_dynamic_sub,
                                         EEG_weak_pitch_static_sub],
                                        file)

                EEG_no_pitch.append(EEG_no_pitch_sub)
                EEG_strong_pitch_dynamic.append(EEG_strong_pitch_dynamic_sub)
                EEG_strong_pitch_static.append(EEG_strong_pitch_static_sub)
                EEG_weak_pitch_dynamic.append(EEG_weak_pitch_dynamic_sub)
                EEG_weak_pitch_static.append(EEG_weak_pitch_static_sub)
                subject.append(dirname)

                # uncomment this in case you want to fix the file by any
                # saved anomaly
                # with open('../EEG_pre_processed.pkl', 'rb') as file_all:
                #     DATA = pickle.load(file_all)
                #     EEG_no_pitch = DATA[0]
                #     EEG_strong_pitch_dynamic = DATA[1]
                #     EEG_strong_pitch_static = DATA[2]
                #     EEG_weak_pitch_dynamic = DATA[3]
                #     EEG_weak_pitch_static = DATA[4]
                #    subject = DATA[5]

                # index_sub = my_list.index('2019-04-15')    # Find the position of 'c' (which is 2)
                # subject.insert(index_sub, dirname)
                # EEG_no_pitch.insert(index_sub, EEG_no_pitch_sub)
                # EEG_strong_pitch_dynamic.insert(index_sub, EEG_strong_pitch_dynamic_sub)
                # EEG_strong_pitch_static.insert(index_sub, EEG_strong_pitch_static_sub)
                # EEG_weak_pitch_dynamic.insert(index_sub, EEG_weak_pitch_dynamic_sub)
                # EEG_weak_pitch_static.insert(index_sub, EEG_weak_pitch_static_sub)

                # save the preprocessed dataset here..
                with open('../EEG_pre_processed.pkl', 'wb') as file_gen:
                    pickle.dump([EEG_no_pitch,
                                 EEG_strong_pitch_dynamic,
                                 EEG_strong_pitch_static,
                                 EEG_weak_pitch_dynamic,
                                 EEG_weak_pitch_static,
                                 subject],
                                file_gen)

                # close the file here..
                file_gen.close()

        # augment subject indexes
        count_subject = count_subject + 1

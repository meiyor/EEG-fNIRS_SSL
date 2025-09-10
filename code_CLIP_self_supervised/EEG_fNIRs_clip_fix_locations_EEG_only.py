"""
This module define the first evaluation of EEG+fNIRs
preliminary exploration on the Steinmetzger et al 2022 dataset
containing EEG+fNIRs multimodal information from simple 1-tone
auditory tasks. We will eventually use this code for extending
The pretrained models coming from this approach to generalized
the models evaluated in NeuraPClab. THIS JUST EXECUTE THE EEG-ONLY MODALITY!!

EEG–fNIRS CLIP-Style Training: EEG-Only Variant

This script implements a training and evaluation pipeline for multimodal contrastive
learning restricted to EEG data only. It includes:

- Dataset preparation and normalization.
- EEG encoder models (EEGNet-inspired, shallow CNNs).
- CLIP-style contrastive loss and supervised fine-tuning.
- Probing utilities: t-SNE, UMAP, classical classifiers.
- Visualization utilities: confusion matrices, relevance maps.

Notes
-----
- This is the EEG-only variant; fNIRS is excluded.
- Shapes: EEG input typically (channels, time), models expect (B, 1, C, T)

"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import mne
import subprocess
import shlex
import h5py
import cv2
import pickle
import pandas as pd
import numpy as np
import random
import torchvision.transforms as T
import torchvision.transforms as transforms
from loguru import logger
from scipy import signal
from mne.io import read_raw_snirf
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Subset
from sklearn.manifold import TSNE

# from openTSNE import TSNE # this was an alternative for scikit-learn TSNE **ALWAYS CONCATENATE THE TRIALS IN A SINGLE TSNE ESTIMATOR PER SUBJECT AS IN THE CODE BELOW***
from sklearn.ensemble import VotingClassifier
from tqdm import tqdm
from typing import Any
from scipy import io
from scipy.stats import wasserstein_distance
from sklearn.model_selection import StratifiedKFold, KFold  # for the stratified k-fold evaluation
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

# use the umap for training - this is replicable and more stable, plot both tsne and umap in this approach
import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns

seed = 42

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# torch.use_deterministic_algorithms(True)


# initializing working seed here
def worker_init_fn(worker_id):
    seed = 42 + worker_id
    np.random.seed(seed)
    random.seed(seed)


# add the Gaussian noise class here
# ---- Custom Transform: Add Gaussian Noise ----
class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean


# define channel values
channels = [
    "Fp1",
    "Fz",
    "F3",
    "F7",
    "FT9",
    "FC5",
    "FC1",
    "C3",
    "T7",
    "TP9",
    "CP5",
    "CP1",
    "Pz",
    "P3",
    "PO9",
    "O1",
    "Oz",
    "O2",
    "PO10",
    "P4",
    "CP6",
    "CP2",
    "Cz",
    "C4",
    "T8",
    "FT10",
    "FC6",
    "FC2",
    "F4",
    "F8",
    "Fp2",
    "AF3",
    "AFz",
    "F1",
    "F5",
    "F9",
    "FT7",
    "FC3",
    "C1",
    "C5",
    "CP3",
    "P9",
    "PO7",
    "POz",
    "I1",
    "FCz",
    "I2",
    "PO8",
    "CPz",
    "CP4",
    "P10",
    "C6",
    "C2",
    "FC4",
    "FT8",
    "F10",
    "F6",
    "F2",
    "AF4",
]


def apply_butter_worth_filter(data: Any, fs: int):
    """
     Apply a zero-phase Butterworth low-pass filter to the data.

     Band-pass filter EEG with a zero-phase Butterworth filter (7–13 Hz).

     Parameters
     ----------
    data : array-like (ndim=1 or 2)
     EEG time-series. When 2D, shape is (n_samples, n_channels) or
       (n_channels, n_samples); filtfilt is applied along the last axis.
      fs : int Sampling frequency (Hz).

      Returns
      -------
      np.ndarray
        Filtered signal with the same shape as `data`.

      Notes
      -----
      - Uses 5th-order band-pass with filtfilt to avoid phase distortion.
      - If your data is shaped (channels, time), ensure time is the last axis
        or pass a view where time is last before calling.
    """

    # filter band-pass
    nyquist = 0.5 * fs
    low = 7 / nyquist
    high = 13 / nyquist

    # Normalize the frequencies for bandpass
    Wn = [low, high]

    # Design the Butterworth filter as bandpass avoid adding distorted phases shifts to the signal
    b, a = signal.butter(5, Wn, btype="bandpass")

    # Apply the filter using filtfilt for zero-phase filtering
    output = signal.filtfilt(b, a, data)

    return output


def reading_ground_truth_index(index_ground_truth_positive, index_ground_truth_negative, folder_data, subject):
    """
    Here do the evaluation of the hbo positive and hbo negative based on the cronological
    order following the Kurt Steimetzger evaluation as groundtruth..
    do the evaluation. This is for getting the HbO pos/neg condition out of the
    calculated median split

    Map ground-truth trial indices (1-based) to the dataset order on disk.

     Parameters
      ----------
      index_ground_truth_positive : list[int]
        1-based indices of HBO-positive trials.
      index_ground_truth_negative : list[int]
        1-based indices of HBO-negative trials.
      folder_data : str
        Path to the subject folder that contains the "EEG data" subdir.
      subject : list[str]
          Chronologically sorted trial folder names for the same subject.

      Returns
      -------
      (np.ndarray, np.ndarray)
       Tuple of (indices_hbo_positive, indices_hbo_negative) as 0-based
       indices into `subject`.

      Raises
      ------
      subprocess.CalledProcessError
        If directory listing fails.
    """

    # get the real indices
    cmd = f'ls -l "{folder_data}/EEG data"'
    out = subprocess.check_output(shlex.split(cmd), text=True)

    dir_names = []
    for line in out.splitlines():
        parts = line.split(None, 8)  # up to 9 fields; name is last
        if len(parts) < 9:
            continue
        mode = parts[0]
        name = parts[-1]
        if mode.startswith("d"):  # directory
            dir_names.append(name)

    indices_hbo_positive = np.array(index_ground_truth_positive) - 1
    indices_hbo_negative = np.array(index_ground_truth_negative) - 1
    folders_positive_ground_truth = [dir_names[i] for i in indices_hbo_positive]
    folders_negative_ground_truth = [dir_names[i] for i in indices_hbo_negative]

    index_HBO_positive = []
    index_HBO_negative = []
    for index_folder in range(0, len(folders_positive_ground_truth)):
        index_HBO_positive.append(subject.index(folders_positive_ground_truth[index_folder]))
        index_HBO_negative.append(subject.index(folders_negative_ground_truth[index_folder]))

    indices_hbo_positive = np.array(index_HBO_positive)
    indices_hbo_negative = np.array(index_HBO_negative)

    return indices_hbo_positive, indices_hbo_negative


def load_matlab_file(path):
    """
    Load a .mat file, v7 or earlier via scipy.io.loadmat,
    or v7.3+ via h5py.
    Parameters:
      path (str): Filesystem path to the .mat file.

    Returns:
      Dict[str, Any]: A dictionary mapping variable names to NumPy arrays.

    Raises:
      FileNotFoundError: If the given path does not exist.
    """

    if not os.path.isfile(path):
        raise FileNotFoundError(f"No such file: {path}")

    # Try HDF5 open first for v7.3+ files
    try:
        with h5py.File(path, "r") as f:
            # if this succeeds, read all datasets into numpy arrays
            return {key: f[key][()] for key in f.keys()}
    except (OSError, IOError):
        # Not an HDF5 file, fall back to scipy.loadmat
        mat = io.loadmat(path)
        # remove MATLAB meta-vars if you like
        for m in ("__header__", "__version__", "__globals__"):
            mat.pop(m, None)
        return mat


def load_info_from_file(data_folder: str):
    """
    get here the loaded info from the .mat file from

    Construct an MNE Info object from EEG data stored in a MATLAB structure.

    This function reads channel labels and trial info from the loaded .mat
    structure, removes non-EEG channels, applies a standard montage,
    and returns a configured mne.Info instance.

    Parameters:
        data_folder (str): Path to the .mat file containing EEG data structure.

    Returns:
        mne.Info: MNE Info object with channel names, sampling frequency, and montage set.
    """
    data_EEG = load_matlab_file(path=data_folder)
    # get the information related to each
    data_EEG["data"]["trialinfo"][0][0].shape[0]
    channels = data_EEG["data"]["label"][0][0]

    CHANNELS = []
    for index_chan in range(0, channels.shape[0]):
        CHANNELS.append(channels[index_chan][0][0])

    # remove the non-existent channels
    removed_indices = [CHANNELS.index("HEOG1"), CHANNELS.index("HEOG2"), CHANNELS.index("VEOG1"), CHANNELS.index("VEOG2")]
    for idx_rm in sorted(removed_indices, reverse=True):
        CHANNELS.pop(idx_rm)

    CHANNELS[CHANNELS.index("PO7'")] = "PO7"
    CHANNELS[CHANNELS.index("PO8'")] = "PO8"

    # get here the montage for each mne raw element
    montage = mne.channels.make_standard_montage("standard_1005")
    all_positions = montage.get_positions()["ch_pos"]
    selected_positions = [all_positions[ch] for ch in CHANNELS if ch in all_positions]

    # get the montage for the new type of mne array
    new_montage = mne.channels.make_dig_montage(ch_pos=dict(zip(CHANNELS, selected_positions)))

    info = mne.create_info(
        ch_names=CHANNELS,
        sfreq=256,  # sampling frequency is 256 by default from the .mat file
        ch_types="eeg",
    )

    info.set_montage(new_montage)

    return info


# plotting function here...
def plotting_relevance_map_eeg_fnirs(eeg_xai_channels, info, epoch, folder_imgs):
    """
    plot here the EEG topo-plot
    """

    # plot here the topoplots for between 350-420 ms
    fig, ax = plt.subplots(figsize=(10, 10))
    im, _ = mne.viz.plot_topomap(np.mean(eeg_xai_channels[:, 178:215], axis=1), info, axes=ax, vlim=(0, 1), names=info.ch_names, cmap="RdBu_r", show=False, sensors=True)

    # ax.set_title(f"{epochs}", fontsize=20)
    clb = plt.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05)

    clb.set_label(label="relevance", size=18)
    clb.ax.tick_params(labelsize=18)
    for text in ax.texts:
        text.set_fontsize(16)  # <-- increase font size her
    plt.suptitle(f"Epoch {epoch}", fontsize=20)
    plt.savefig(f"{folder_imgs}/relevance_measure_{epoch}_350_400.jpg", dpi=300)

    plt.close("all")

    # plot here the topoplots for between 350-420 ms
    fig, ax = plt.subplots(figsize=(10, 10))
    im, _ = mne.viz.plot_topomap(np.mean(eeg_xai_channels[:, 25:51], axis=1), info, axes=ax, vlim=(0, 1), names=info.ch_names, cmap="RdBu_r", show=False, sensors=True)

    clb = plt.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05)

    clb.set_label(label="relevance", size=18)
    clb.ax.tick_params(labelsize=18)

    for text in ax.texts:
        text.set_fontsize(16)  # <-- increase font size her
    plt.suptitle(f"Epoch {epoch}", fontsize=20)
    plt.savefig(f"{folder_imgs}/relevance_measure_{epoch}_50_100.jpg", dpi=300)

    plt.close("all")

    # plot here the topoplots for between 350-420 ms
    fig, ax = plt.subplots(figsize=(10, 10))
    im, _ = mne.viz.plot_topomap(np.mean(eeg_xai_channels[:, :], axis=1), info, axes=ax, vlim=(0, 1), names=info.ch_names, cmap="RdBu_r", show=False, sensors=True)

    clb = plt.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05)

    clb.set_label(label="relevance", size=18)
    clb.ax.tick_params(labelsize=18)
    for text in ax.texts:
        text.set_fontsize(16)  # <-- increase font size her
    plt.suptitle(f"Epoch {epoch}", fontsize=20)
    plt.savefig(f"{folder_imgs}/relevance_measure_{epoch}.jpg", dpi=300)

    plt.close("all")


def init_weights_uniform(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        torch.nn.init.uniform_(m.weight, a=-0.1, b=0.1)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


def init_weights_xavier(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


# define here the normalization functions
def normalization_trial_minmax(x):
    min_val = x.min()
    max_val = x.max()
    return (x - min_val) / (max_val - min_val)


def normalization_trial_zscore(x):
    mean = x.mean()
    std = x.std()
    return (x - mean) / (std)


# define random erase across time with random ROIs patches FC_R1, FC_R2, CP_1, CP_2, P_1, P_2, O_1 with overlap. This is defined for a coarse grained ROI-based occlusion methods.
class RandomErasingOneROIs:
    def __init__(self, p=0.5, min_frac=0.1, max_frac=0.5, value=0.0, channels=None):
        """
        erase_along: 'height' or 'width'
        p: probability of applying
        value: fill value for erased region
        """
        self.p = p
        self.min_frac = min_frac
        self.max_frac = max_frac
        self.value = value
        self.channels = channels

    def __call__(self, x):
        if random.random() > self.p:
            return x  # skip

        x = x.clone()
        b, channels, time = x.shape

        # select the ROI randomly
        roi_selector = random.randint(0, 6)

        erase_len = random.randint(int(time * self.min_frac), int(time * self.max_frac))
        start = random.randint(0, time - erase_len)

        if roi_selector == 0:
            indexes_rem = [
                self.channels.index("F5"),
                self.channels.index("F3"),
                self.channels.index("F1"),
                self.channels.index("Fz"),
                self.channels.index("F2"),
                self.channels.index("F4"),
                self.channels.index("F6"),
                self.channels.index("FC5"),
                self.channels.index("FC3"),
                self.channels.index("FC1"),
                self.channels.index("FCz"),
                self.channels.index("FC2"),
                self.channels.index("FC4"),
                self.channels.index("FC6"),
            ]
        elif roi_selector == 1:
            indexes_rem = [
                self.channels.index("FC5"),
                self.channels.index("FC3"),
                self.channels.index("FC1"),
                self.channels.index("FCz"),
                self.channels.index("FC2"),
                self.channels.index("FC4"),
                self.channels.index("FC6"),
                self.channels.index("C5"),
                self.channels.index("C3"),
                self.channels.index("C1"),
                self.channels.index("Cz"),
                self.channels.index("C2"),
                self.channels.index("C4"),
                self.channels.index("C6"),
            ]
        elif roi_selector == 2:
            indexes_rem = [
                self.channels.index("C5"),
                self.channels.index("C3"),
                self.channels.index("C1"),
                self.channels.index("Cz"),
                self.channels.index("C2"),
                self.channels.index("C4"),
                self.channels.index("C6"),
                self.channels.index("CP5"),
                self.channels.index("CP3"),
                self.channels.index("CP1"),
                self.channels.index("CPz"),
                self.channels.index("CP2"),
                self.channels.index("CP4"),
                self.channels.index("CP6"),
            ]
        elif roi_selector == 3:
            indexes_rem = [
                self.channels.index("CP5"),
                self.channels.index("CP3"),
                self.channels.index("CP1"),
                self.channels.index("CPz"),
                self.channels.index("CP2"),
                self.channels.index("CP4"),
                self.channels.index("CP6"),
                self.channels.index("P4"),
                self.channels.index("P3"),
                self.channels.index("Pz"),
            ]
        elif roi_selector == 4:
            indexes_rem = [
                self.channels.index("P3"),
                self.channels.index("Pz"),
                self.channels.index("P4"),
                self.channels.index("POz"),
                self.channels.index("PO7"),
                self.channels.index("PO8"),
                self.channels.index("O1"),
                self.channels.index("Oz"),
                self.channels.index("O2"),
            ]
        elif roi_selector == 5:
            indexes_rem = [
                self.channels.index("PO8"),
                self.channels.index("O1"),
                self.channels.index("Oz"),
                self.channels.index("O2"),
                self.channels.index("PO7"),
                self.channels.index("P9"),
                self.channels.index("PO9"),
                self.channels.index("P10"),
                self.channels.index("PO10"),
            ]
        elif roi_selector == 6:
            indexes_rem = [self.channels.index("P9"), self.channels.index("PO9"), self.channels.index("P10"), self.channels.index("PO10"), self.channels.index("I2"), self.channels.index("I1")]
        x[:, indexes_rem, start : start + erase_len] = self.value

        return x


# define random erase across time with random ROIs p
class RandomErasingOneROIsMoreGranular:
    def __init__(self, p=0.5, min_frac=0.1, max_frac=0.5, value=0.0, channels=None):
        """
        erase_along: 'height' or 'width'
        p: probability of applying
        value: fill value for erased region
        """
        self.p = p
        self.min_frac = min_frac
        self.max_frac = max_frac
        self.value = value
        self.channels = channels

    def __call__(self, x):
        if random.random() > self.p:
            return x  # skip

        x = x.clone()
        b, channels, time = x.shape

        # select the ROI randomly
        # roi_selector = random.randint(0, 27)
        population = list(range(0, 28))

        # Get a sample of unique random integers
        random_numbers = random.sample(population, 3)

        erase_len = random.randint(int(time * self.min_frac), int(time * self.max_frac))
        start = random.randint(0, time - erase_len)

        index_list = []

        for index_roi in range(0, 3):
            roi_selector = random_numbers[index_roi]

            if roi_selector == 0:
                indexes_rem = [self.channels.index("F5"), self.channels.index("F3"), self.channels.index("FC5"), self.channels.index("FC3")]
            elif roi_selector == 1:
                indexes_rem = [self.channels.index("F3"), self.channels.index("FC3"), self.channels.index("F1"), self.channels.index("FC1")]
            elif roi_selector == 2:
                indexes_rem = [self.channels.index("F1"), self.channels.index("FC1"), self.channels.index("Fz"), self.channels.index("FCz")]
            elif roi_selector == 3:
                indexes_rem = [self.channels.index("Fz"), self.channels.index("FCz"), self.channels.index("F2"), self.channels.index("FC2")]
            elif roi_selector == 4:
                indexes_rem = [self.channels.index("F2"), self.channels.index("FC2"), self.channels.index("F4"), self.channels.index("FC4")]
            elif roi_selector == 5:
                indexes_rem = [self.channels.index("F4"), self.channels.index("FC4"), self.channels.index("F6"), self.channels.index("FC6")]
            elif roi_selector == 6:
                indexes_rem = [self.channels.index("FC5"), self.channels.index("FC3"), self.channels.index("C3"), self.channels.index("C5")]
            elif roi_selector == 7:
                indexes_rem = [self.channels.index("FC3"), self.channels.index("FC1"), self.channels.index("C3"), self.channels.index("C1")]
            elif roi_selector == 8:
                indexes_rem = [self.channels.index("FC1"), self.channels.index("FCz"), self.channels.index("C1"), self.channels.index("Cz")]
            elif roi_selector == 9:
                indexes_rem = [self.channels.index("FCz"), self.channels.index("FC2"), self.channels.index("Cz"), self.channels.index("C2")]
            elif roi_selector == 10:
                indexes_rem = [self.channels.index("FC4"), self.channels.index("FC2"), self.channels.index("C2"), self.channels.index("C4")]
            elif roi_selector == 11:
                indexes_rem = [self.channels.index("FC4"), self.channels.index("FC6"), self.channels.index("C4"), self.channels.index("C6")]
            elif roi_selector == 12:
                indexes_rem = [self.channels.index("C5"), self.channels.index("C3"), self.channels.index("CP5"), self.channels.index("CP3")]
            elif roi_selector == 13:
                indexes_rem = [self.channels.index("C3"), self.channels.index("CP3"), self.channels.index("C1"), self.channels.index("CP1")]
            elif roi_selector == 14:
                indexes_rem = [self.channels.index("Cz"), self.channels.index("C2"), self.channels.index("CPz"), self.channels.index("CP2")]
            elif roi_selector == 15:
                indexes_rem = [self.channels.index("C2"), self.channels.index("C4"), self.channels.index("CP2"), self.channels.index("CP4")]
            elif roi_selector == 16:
                indexes_rem = [self.channels.index("C4"), self.channels.index("C6"), self.channels.index("CP4"), self.channels.index("CP6")]
            elif roi_selector == 17:
                indexes_rem = [self.channels.index("CP5"), self.channels.index("CP3"), self.channels.index("P3"), self.channels.index("Pz")]
            elif roi_selector == 18:
                indexes_rem = [self.channels.index("CP1"), self.channels.index("CPz"), self.channels.index("Pz")]
            elif roi_selector == 19:
                indexes_rem = [self.channels.index("P3"), self.channels.index("Pz"), self.channels.index("POz")]
            elif roi_selector == 20:
                indexes_rem = [self.channels.index("Pz"), self.channels.index("P4"), self.channels.index("POz")]
            elif roi_selector == 21:
                indexes_rem = [self.channels.index("POz"), self.channels.index("O1"), self.channels.index("Oz"), self.channels.index("O2")]
            elif roi_selector == 22:
                indexes_rem = [self.channels.index("PO7"), self.channels.index("O1"), self.channels.index("POz"), self.channels.index("Oz")]
            elif roi_selector == 23:
                indexes_rem = [self.channels.index("POz"), self.channels.index("Oz"), self.channels.index("O2"), self.channels.index("PO8")]
            elif roi_selector == 24:
                indexes_rem = [self.channels.index("O1"), self.channels.index("Oz"), self.channels.index("I1"), self.channels.index("PO9")]
            elif roi_selector == 25:
                indexes_rem = [self.channels.index("Oz"), self.channels.index("O2"), self.channels.index("I2"), self.channels.index("PO10")]
            elif roi_selector == 26:
                indexes_rem = [self.channels.index("P9"), self.channels.index("PO7"), self.channels.index("O1"), self.channels.index("PO9")]
            elif roi_selector == 27:
                indexes_rem = [self.channels.index("PO8"), self.channels.index("PO10"), self.channels.index("O2"), self.channels.index("P10")]

            index_list.extend(indexes_rem)

        x[:, np.unique(np.array(index_list)), start : start + erase_len] = self.value

        return x


# define the random erase for always remove all the channels
class RandomErasingOneDimPatch:
    def __init__(self, erase_along="height", p=0.5, min_frac=0.1, max_frac=0.5, value=0.0):
        """
        erase_along: 'height' or 'width'
        p: probability of applying
        value: fill value for erased region
        """
        self.erase_along = erase_along
        self.p = p
        self.min_frac = min_frac
        self.max_frac = max_frac
        self.value = value

    def __call__(self, x):
        # x: (C, H, W) for images or (C, T) for EEG/time-series
        if random.random() > self.p:
            return x  # skip

        x = x.clone()
        b, channels, time = x.shape

        if self.erase_along == "width":
            # Pick a random row index
            erase_len = random.randint(int(time * self.min_frac), int(time * self.max_frac))
            start = random.randint(0, time - erase_len)
            x[:, :, start : start + erase_len] = self.value
        elif self.erase_along == "height":
            # Pick a random column index
            erase_len = random.randint(int(channels * self.min_frac), int(channels * self.max_frac))
            start = random.randint(0, channels - erase_len)
            x[:, start : start + erase_len, :] = self.value
        else:
            raise ValueError("erase_along must be 'height' or 'width'")

        return x


# plot the confusion matrix here after the finetuning evaluation
def plot_confusion_matrix_test(labels: Any, preds: Any, labels_disp: Any, suffix: str, title: str, metrics: Any, folders: str):
    """
    plot here the confusion matrix here..
    """

    # Example: assuming y_trues, y_scores, labels_int, labels_disp
    CMatrix = confusion_matrix(labels, preds)
    disp_cmatrix = ConfusionMatrixDisplay(confusion_matrix=CMatrix, display_labels=labels_disp)

    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(10, 10))
    disp_cmatrix.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False, values_format="", include_values=False)

    # Calculate total trials for percentage
    total_trials = np.sum(CMatrix)

    # Annotate each cell with "count (rate%)"
    for i in range(CMatrix.shape[0]):
        for j in range(CMatrix.shape[1]):
            count = CMatrix[i, j]
            rate = (count / total_trials) * 100 if total_trials > 0 else 0
            ax.text(j, i, f"{count}\n({rate:.1f}%)", ha="center", va="center", color="white" if CMatrix[i, j] > CMatrix.max() / 2 else "black", fontsize=12)

    # Set title and save
    ax.set_title(f"{title} metrics: pr={metrics[1]}, re={metrics[2]},\n F1={metrics[3]}, acc={metrics[0]}")
    plt.tight_layout()
    plt.savefig(f"{folders}/conf_matrix_epoch_fold_{suffix}.jpg", dpi=300)
    plt.close("all")


# plotting tsne features
def plot_tsne_feat(tsne_feat, labels, title: str, epoch: str, suffix: str, folder_name: str, class_value: str, subj: str):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=tsne_feat[:, 0], y=tsne_feat[:, 1], hue=labels, palette="tab10", s=60, edgecolor="k", alpha=0.7)
    plt.title(title)
    plt.legend(title=class_value, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(f"{folder_name}/epoch_{epoch}_{suffix}_{subj}.jpg")
    plt.close("all")


# do here the plotting twinx to show two different variables measurements here
def plotting_twinx_variables(time_vector: Any, data1: Any, data2: Any, title: str, x_label: str, y_label1: str, y_label2: str, folder_images: str, subj: str):
    """
    plot her twinx the variables you want to compare
    across number of epochs in this case
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(time_vector, data1, "b-", label=y_label1, linewidth=3)
    ax1.set_xlabel(x_label, fontsize=1)
    ax1.set_ylabel(y_label1, color="blue", fontsize=16)
    ax1.tick_params(axis="y", labelcolor="blue")

    ax2 = ax1.twinx()

    ax2.plot(time_vector, data2, "r-", label=y_label2, linewidth=3)
    ax2.set_ylabel(y_label2, color="red", fontsize=16)
    ax2.tick_params(axis="y", labelcolor="red")

    ax1.grid(True)

    for tick in ax1.get_xticklabels():
        tick.set_fontsize(14)

    plt.title(title)
    fig.legend()
    fig.savefig(f"{folder_images}/{y_label1}_{y_label2}_{subj}.jpg")
    plt.close("all")


# set here the dataset definition using a CLIP organization between pairs
class CLIPDataset(Dataset):
    """
    Torch dataset returning (EEG, fNIRS, label, subject_id, HbO_label).

    Parameters
    ----------
    eeg : np.ndarray, shape (N, C_eeg, T_eeg)
    fnirs : np.ndarray, shape (N, C_fnirs, T_fnirs)
    labels : np.ndarray, shape (N,)
       Stimulus/category ids.
    subject_labels : np.ndarray, shape (N,)
       Subject ids (ints).
    labels_HbO : np.ndarray, shape (N,)
      HbO labels (e.g., 0/1).
    normalization_function : Callable[[np.ndarray], np.ndarray]
      Per-trial normalization (e.g., minmax or z-score).

    __getitem__(idx) -> tuple[torch.FloatTensor, torch.FloatTensor, float, float, float]
     EEG tensor (H,W), fNIRS tensor (H,W), label, subject_label, hbo_label.
    """

    def __init__(self, eeg, fnirs, labels, subject_labels, labels_HbO, normalization_function, eeg_transform=None, fnirs_transform=None):
        assert len(eeg) == len(fnirs), "Mismatch in image/trial count"
        self.eeg = eeg
        self.fnirs = fnirs  # NumPy array: (trials, height, width)
        self.labels = labels.astype(float)
        self.subject_labels = subject_labels.astype(float)
        self.labels_hbo = labels_HbO.astype(float)
        self.normalization_function = normalization_function

        # transforms
        self.eeg_transform = eeg_transform
        self.fnirs_transform = fnirs_transform

    def __len__(self):
        return len(self.eeg)

    def __getitem__(self, idx):
        eeg_img = self.eeg[idx]
        fnirs_img = self.fnirs[idx]  # shape: (height, width)
        labels_output = self.labels[idx]
        labels_subject = self.subject_labels[idx]
        labels_hbo = self.labels_hbo[idx]

        eeg_img = self.normalization_function(eeg_img)
        fnirs_img = self.normalization_function(fnirs_img)

        # Convert EEG to tensor
        eeg_tensor = torch.tensor(eeg_img, dtype=torch.float32)  # shape: (H, W)
        fnirs_tensor = torch.tensor(fnirs_img, dtype=torch.float32)  # shape: (H, W)

        # get tge tranformations here
        if self.eeg_transform:
            eeg_tensor_trans = self.eeg_transform(eeg_tensor.unsqueeze(0))
        if self.fnirs_transform:
            fnirs_tensor_trans = self.fnirs_transform(fnirs_tensor.unsqueeze(0))

        return eeg_tensor, fnirs_tensor, labels_output, labels_subject, labels_hbo, eeg_tensor_trans.squeeze(), fnirs_tensor_trans.squeeze()


# define here the fNIRs_encoder as EEGNet variation
class fNIRsNet(nn.Module):
    """
    EEGNet-inspired encoder given for fNIRS, with attention Multi-head.

    Parameters
    ----------
    n_channels : int, default=80
    embed_dim : int, default=128
    n_times : int, default=9
    dropout, F1, D, F2 : hyperparameters

     Forward
    -------
    forward(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]
     x: (B, 1, C, T)
    Returns (embedding, flattened_conv_features).
    """

    def __init__(self, n_channels=80, embed_dim=128, n_times=9, dropout=0.25, F1=8, D=2, F2=16):
        super(fNIRsNet, self).__init__()
        self.firstconv = nn.Sequential(nn.Conv2d(1, F1, [3, 1], padding=0, bias=False), nn.BatchNorm2d(F1))
        self.depthwiseConv = nn.Sequential(nn.Conv2d(F1, F1 * D, [2, 1], groups=F1, bias=False), nn.BatchNorm2d(F1 * D), nn.ELU(), nn.AvgPool2d((1, 1)), nn.Dropout(dropout))
        self.separableConv = nn.Sequential(nn.Conv2d(F1 * D, F2, [1, 1], padding=0, bias=False), nn.BatchNorm2d(F2), nn.ELU(), nn.AvgPool2d((1, 1)), nn.Dropout(dropout))

        self.flatten = nn.Flatten()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=2, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

        # You can compute the final feature dimension dynamically
        # Compute flatten_dim using a dummy forward pass
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, n_channels, n_times)  # shape: (B, 1, C, T)
            out = self.firstconv(dummy_input)
            out = self.depthwiseConv(out)
            out = self.separableConv(out)
            self.flatten_dim = out.view(1, -1).size(1)

        # get the flattenn from here
        self.head = nn.Sequential(
            nn.Linear(self.flatten_dim, embed_dim),
        )

    def forward(self, x):  # x shape: (B, 1, C, T)
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = self.flatten(x)
        x_flatten = x
        x = self.head(x)
        attn_output, _ = self.attn(x, x, x)
        x = self.norm(attn_output + x)
        return x, x_flatten


# define here the EEG_encoder as EEGNet
class EEGNet(nn.Module):
    def __init__(self, n_channels=43, embed_dim=128, n_times=548, dropout=0.25, F1=8, D=2, F2=16):
        super(EEGNet, self).__init__()
        self.firstconv = nn.Sequential(nn.Conv2d(1, F1, [1, n_channels], padding=0, bias=False), nn.BatchNorm2d(F1))
        self.depthwiseConv = nn.Sequential(nn.Conv2d(F1, F1 * D, [round(n_channels / 2), 1], groups=F1, bias=False), nn.BatchNorm2d(F1 * D), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(dropout))
        self.separableConv = nn.Sequential(nn.Conv2d(F1 * D, F2, [1, 16], padding=0, bias=False), nn.BatchNorm2d(F2), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(dropout))

        self.flatten = nn.Flatten()

        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=2, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

        # You can compute the final feature dimension dynamically
        # Compute flatten_dim using a dummy forward pass
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, n_channels, n_times)  # shape: (B, 1, C, T)
            out = self.firstconv(dummy_input)
            out = self.depthwiseConv(out)
            out = self.separableConv(out)
            self.flatten_dim = out.view(1, -1).size(1)

        # get the flattenn from here
        self.head = nn.Sequential(
            nn.Linear(self.flatten_dim, embed_dim),
        )

    def forward(self, x):  # x shape: (B, 1, C, T)
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = self.flatten(x)
        x_flatten = x
        x = self.head(x)
        attn_output, _ = self.attn(x, x, x)
        x = self.norm(attn_output + x)
        return x, x_flatten


# define here the head class
class AttHead(nn.Module):
    def __init__(self, flatten_dim, embed_dim=128):
        super(AttHead, self).__init__()

        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=2, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

        # get the flattenn from here
        self.head = nn.Sequential(
            nn.Linear(flatten_dim, embed_dim),
        )

    def forward(self, x):  # x shape: (B, 1, C, T)
        x = self.head(x)
        attn_output, _ = self.attn(x, x, x)
        x = self.norm(attn_output + x)
        return x


# define here the fine-tune model as a full class
class EEGNetFineTuned(nn.Module):
    def __init__(self, EEGNet_extractor, fNIRsNet_extractor, embed_dim=128, label_dim=2):  # replace with the number of labels for initial evaluation
        super(EEGNetFineTuned, self).__init__()
        self.feature_extractor = EEGNet_extractor
        self.feature_extractor_fnirs = fNIRsNet_extractor
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_extractor.flatten_dim + self.feature_extractor_fnirs.flatten_dim, label_dim),
            nn.ELU(),
            # nn.Linear(embed_dim, label_dim),
        )
        # self.flatten = nn.Flatten()
        # don't use this for the fine tune part just leave it for the self-supervised section
        # self.linear = nn.Linear(self.feature_extractor.flatten_dim, embed_dim) #+ self.feature_extractor_fnirs.flatten_dim, embed_dim)
        # self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=2, batch_first=True)
        # self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x_eeg, x_fnirs):
        x_eeg, _ = self.feature_extractor(x_eeg)
        x_fnirs, _ = self.feature_extractor_fnirs(x_fnirs)
        # x = self.linear(x_eeg)
        # x = self.flatten(x)
        # attn_output, _ = self.attn(x, x, x)
        # x = self.norm(attn_output + x)
        x = self.classifier(torch.cat((x_eeg, x_fnirs), 1))
        return x


# define the ELU classifier here
class ReLUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(ReLUClassifier, self).__init__()
        # One hidden layer + output layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # fully connected
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)  # output layer

    def forward(self, x_eeg, x_fnirs):
        x = self.fc1(torch.cat((x_eeg, x_fnirs), 1))  # input -> hidden
        x = self.elu(x)  # apply ReLU
        # don't add this to prevent overfitting..
        # x = self.fc2(x)  # hidden -> output
        return x


# define here the weight activation
def weight_activation(feats, grads):
    cam = feats * F.relu(grads)
    cam = torch.sum(cam, dim=1).squeeze().cpu().detach().numpy()
    return cam


# define the GradCAM class here
class GradCAM(nn.Module):
    def __init__(self, encoder_EEG, encoder_fNIRs, attentional_head_eeg, attentional_head_fnirs):
        super(GradCAM, self).__init__()
        self.gradients = {}
        self.features = {}

        self.feature_extractor_eeg = encoder_EEG
        self.feature_extractor_fnirs = encoder_fNIRs

        self.contrastive_head_eeg = attentional_head_eeg
        self.contrastive_head_fnirs = attentional_head_fnirs

        self.measure = nn.CosineSimilarity(dim=-1)
        self.flatten = nn.Flatten()

    def save_grads(self, img_index):
        def hook(grad):
            self.gradients[img_index] = grad.detach()

        return hook

    def save_features(self, img_index, feats):
        self.features[img_index] = feats.detach()

    def forward(self, eeg_img, fnirs_img):
        _, features_eeg = self.feature_extractor_eeg(eeg_img)
        _, features_fnirs = self.feature_extractor_fnirs(fnirs_img)

        self.save_features("EEG", features_eeg)
        self.save_features("fNIRs", features_fnirs)

        h1 = features_eeg.register_hook(self.save_grads("EEG"))
        h2 = features_fnirs.register_hook(self.save_grads("fNIRs"))

        out_eeg, out_fnirs = self.contrastive_head_eeg(self.flatten(features_eeg)), self.contrastive_head_fnirs(self.flatten(features_fnirs))
        score = self.measure(out_eeg, out_fnirs)

        h1.remove()
        h2.remove()

        return score


# define here the feature extractor without the head
class EEGNetFeatureExtractor(nn.Module):
    def __init__(self, n_channels=43, n_times=548, dropout=0.25, F1=8, D=2, F2=16):
        super(EEGNetFeatureExtractor, self).__init__()
        self.firstconv = nn.Sequential(nn.Conv2d(1, F1, [1, n_channels], padding=0, bias=False), nn.BatchNorm2d(F1))
        self.depthwiseConv = nn.Sequential(nn.Conv2d(F1, F1 * D, [round(n_channels / 2), 1], groups=F1, bias=False), nn.BatchNorm2d(F1 * D), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(dropout))
        self.separableConv = nn.Sequential(nn.Conv2d(F1 * D, F2, [1, 16], padding=0, bias=False), nn.BatchNorm2d(F2), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(dropout))
        self.flatten = nn.Flatten()

        # Calculate flatten dim
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, n_channels, n_times)
            out = self.firstconv(dummy_input)
            out = self.depthwiseConv(out)
            out = self.separableConv(out)
            self.flatten_dim = out.view(1, -1).size(1)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x_prev = x
        x = self.flatten(x)
        return x, x_prev


# define here the fNIRs_encoder as EEGNet
class fNIRsNetFeatureExtractor(nn.Module):
    def __init__(self, n_channels=80, n_times=9, dropout=0.25, F1=8, D=2, F2=16):
        super(fNIRsNetFeatureExtractor, self).__init__()
        self.firstconv = nn.Sequential(nn.Conv2d(1, F1, [3, 1], padding=0, bias=False), nn.BatchNorm2d(F1))
        self.depthwiseConv = nn.Sequential(nn.Conv2d(F1, F1 * D, [2, 1], groups=F1, bias=False), nn.BatchNorm2d(F1 * D), nn.ELU(), nn.AvgPool2d((1, 1)), nn.Dropout(dropout))
        self.separableConv = nn.Sequential(nn.Conv2d(F1 * D, F2, [1, 1], padding=0, bias=False), nn.BatchNorm2d(F2), nn.ELU(), nn.AvgPool2d((1, 1)), nn.Dropout(dropout))

        self.flatten = nn.Flatten()

        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, n_channels, n_times)  # shape: (B, 1, C, T)
            out = self.firstconv(dummy_input)
            out = self.depthwiseConv(out)
            out = self.separableConv(out)
            self.flatten_dim = out.view(1, -1).size(1)

    def forward(self, x):  # x shape: (B, 1, C, T)
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x_prev = x
        x = self.flatten(x)
        return x, x_prev


# define here the encoders we will use for testing, let's check first a ShallowNet here
class ShallowConvNet_EEG(nn.Module):
    def __init__(self, input_shape, embed_dim=512, use_norm=True):
        """
        input_shape: (C, H, W)
        """
        super().__init__()
        C, H, W = input_shape
        self.use_norm = use_norm

        def block(in_ch, out_ch, conv_size, pool_size):
            return nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=conv_size, padding=1), nn.BatchNorm2d(out_ch) if use_norm else nn.Identity(), nn.ReLU(), nn.MaxPool2d(kernel_size=pool_size))

        self.encoder = nn.Sequential(block(C, 16, [10, 5], [2, 5]), block(16, 128, [5, 5], [2, 2]))

        # Compute output size after conv layers
        dummy = torch.zeros(1, C, H, W)
        out = self.encoder(dummy)
        flatten_dim = out.view(1, -1).size(1)

        # get the flattenn from here
        self.head = nn.Sequential(
            nn.Linear(flatten_dim, embed_dim),
            # nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return x


class ShallowConvNet_fNIRs(nn.Module):
    def __init__(self, input_shape, embed_dim=512, use_norm=True):
        """
        input_shape: (C, H, W)
        """
        super().__init__()
        C, H, W = input_shape
        self.use_norm = use_norm

        def block(in_ch, out_ch, conv_size, pool_size):
            return nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=conv_size, padding=1), nn.BatchNorm2d(out_ch) if use_norm else nn.Identity(), nn.ReLU(), nn.MaxPool2d(kernel_size=pool_size))

        self.encoder = nn.Sequential(block(C, 16, [5, 2], [5, 1]), block(16, 128, [5, 1], [5, 1]))

        # Compute output size after conv layers
        dummy = torch.zeros(1, C, H, W)
        out = self.encoder(dummy)
        flatten_dim = out.view(1, -1).size(1)

        # get the flattenn from here
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, embed_dim),
            # nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return x


# function for testing the finetune model
def evaluate_finetune_model(trained_model, dataloader, device="cuda", option_training=None, epoch=None):
    """
    Evaluate a fine-tuned classifier on a dataloader.

    Parameters
    ----------
    trained_model : nn.Module
    dataloader : DataLoader
    device : str, default='cuda'
    option_training : {'subject','HbO','stimulus'}
     Which target to supervise/evaluate.
    epoch : Optional[int]
     Carried for logging/filenames elsewhere.

     Returns
     -------
     tuple
     (acc, pr_weighted, re_macro, f1_weighted,
      total_loss, labels_np, preds_np)
    """

    criterion = nn.CrossEntropyLoss()
    trained_model.eval()
    correct = 0
    total = 0
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for eegs, fnirs, labels, subject_labels, labels_hbo in dataloader:
            inputs = eegs.to(device)
            inputs_fnirs = fnirs.to(device)
            labels = labels.to(device).long()
            subject_labels = subject_labels.to(device).long()
            labels_hbo = labels_hbo.to(device).long()

            outputs = trained_model(inputs.unsqueeze(1).float(), inputs_fnirs.unsqueeze(1).float())
            if option_training == "subject":
                loss = criterion(outputs, subject_labels.long())
            elif option_training == "HbO":
                loss = criterion(outputs, labels_hbo.long())
            elif option_training == "stimulus":
                loss = criterion(outputs, labels.long())
            total_loss += loss.item()

            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            if option_training == "subject":
                all_labels.extend(subject_labels.cpu().numpy())
                correct += (preds == subject_labels).sum().item()
                total += subject_labels.size(0)
            elif option_training == "HbO":
                all_labels.extend(labels_hbo.cpu().numpy())
                correct += (preds == labels_hbo).sum().item()
                total += labels_hbo.size(0)
            elif option_training == "stimulus":
                all_labels.extend(labels.cpu().numpy())
                correct += (preds == labels).sum().item()
                total += labels.size(0)

    acc = correct / total
    f1 = f1_score(all_labels, all_preds, average="weighted")
    pr = precision_score(all_labels, all_preds, average="weighted")
    re = recall_score(all_labels, all_preds, average="macro")
    return acc, pr, re, f1, total_loss, np.array(all_labels), np.array(all_preds)


# define the training function for the finetuned model
def train_finetune_model(finetune_model, train_loader, test_loader, epochs=100, device="cuda", option_training=None, epoch_self=None, lr=1e-3):
    """
    Train a fine-tuning classifier with AdamW and cosine annealing; return best state.

    Parameters
    ----------
    finetune_model : nn.Module
    train_loader, test_loader : DataLoader
    epochs : int
    device : str
    option_training : {'subject','HbO','stimulus'}
    epoch_self : Optional[int]
     For passing through to evaluation utilities.
    lr : float
     Initial learning rate.

    Returns
    -------
    tuple
     (best_model, train_acc, train_f1, train_pr, train_re,
      best_test_acc, best_test_f1, best_test_pr, best_test_re,
      epoch_of_best, train_loss_at_best, test_loss_at_best,
      best_optimizer, y_test, y_pred)

    """

    model = finetune_model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)  # , weight_decay=1e-6)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=2e-4)

    average_acc_f1s = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        for eeg_imgs, fnirs_imgs, labels, subject_labels, labels_hbo in tqdm(train_loader, disable=True, desc=f"Epoch {epoch}/{epochs}"):
            inputs = eeg_imgs.to(device)
            inputs_fnirs = fnirs_imgs.to(device)
            labels = labels.to(device)
            subject_labels = subject_labels.to(device)
            labels_hbo = labels_hbo.to(device)

            optimizer.zero_grad()

            # use the two encoder as feature extractors DON'T make it to train..
            outputs = model(inputs.unsqueeze(1).float(), inputs_fnirs.unsqueeze(1).float())
            if option_training == "subject":
                loss = criterion(outputs, subject_labels.long())
            elif option_training == "HbO":
                loss = criterion(outputs, labels_hbo.long())
            elif option_training == "stimulus":
                loss = criterion(outputs, labels.long())

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            if option_training == "subject":
                all_labels.extend(subject_labels.cpu().numpy())
                correct += (preds == subject_labels).sum().item()
                total += subject_labels.size(0)
            elif option_training == "HbO":
                all_labels.extend(labels_hbo.cpu().numpy())
                correct += (preds == labels_hbo).sum().item()
                total += labels_hbo.size(0)
            elif option_training == "stimulus":
                all_labels.extend(labels.cpu().numpy())
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        train_acc = correct / total
        train_f1 = f1_score(all_labels, all_preds, average="weighted")
        train_pr = precision_score(all_labels, all_preds, average="weighted")
        train_re = recall_score(all_labels, all_preds, average="macro")
        test_acc, test_pr, test_re, test_f1, loss_test, labels_t, pred_t = evaluate_finetune_model(model, test_loader, device, option_training, epoch=epoch_self)

        if epoch > 1:
            if np.mean([test_acc, test_f1, test_re]) >= np.max(np.array(average_acc_f1s)):
                max_acc, max_f1, max_pr, max_re = test_acc, test_f1, test_pr, test_re
                train_value_acc, train_value_f1, pr_value, re_value = train_acc, train_f1, train_pr, train_re
                epoch_max = epoch
                loss_opt = total_loss
                loss_test_max = loss_test
                optimizer_max = optimizer
                model_max = model
                labels_test = labels_t
                pred_test = pred_t

        average_acc_f1s.append(np.mean([test_acc, test_f1, test_re]))

        print(f"Epoch {epoch}: Loss={total_loss:.4f}, Train Acc={train_acc:.4f}, Train F1={train_f1:.4f} - Test Acc={test_acc:.4f}, Test F1={test_f1}")

    # return the compiled best accuracies values
    return model_max, train_value_acc, train_value_f1, pr_value, re_value, max_acc, max_f1, max_pr, max_re, epoch_max, loss_opt, loss_test_max, optimizer_max, labels_test, pred_test


# define here the los calculation
def clip_loss(z_a, z_b, temperature=0.07):
    """
    InfoNCE loss between two modality embeddings (z_a, z_b).

    Symmetric InfoNCE-style CLIP loss with cross-entropy over cosine logits.
    This is just the evaluation of the self-supervised part NO labels
    included here.

    Parameters
    ----------
    z_a, z_b : torch.Tensor, shape (B, D)
        Modality embeddings.
    temperature : float
        Softmax temperature.

    Returns
     -------
     torch.Tensor
        Scalar loss

    """

    # Compute cosine similarity
    # logits_per_image: (batch_size, batch_size) where element (i, j) is similarity of eeg_i and fnirs_j
    eeg_fnirs = torch.matmul(z_a, z_b.T) / temperature
    # logits_per_text: (batch_size, batch_size) where element (i, j) is similarity of fnirs_i and eeg_j
    fnirs_eeg = torch.matmul(z_b, z_a.T) / temperature

    # Create labels for cross-entropy loss (diagonal elements represent positive pairs)
    labels = torch.arange(z_a.shape[0], device=z_a.device)

    eeg_loss = F.cross_entropy(eeg_fnirs, labels, reduction="none")
    fnirs_loss = F.cross_entropy(fnirs_eeg, labels, reduction="none")
    loss = (eeg_loss + fnirs_loss) / 2.0  # shape: (batch_size)

    return loss.mean()


def clip_loss_conditional(z_a, z_b, cond_labels, temperature=0.07):
    """
    Supervised contrastive InfoNCE loss for two modalities (EEG=z_a, fNIRS=z_b),
    using condition labels to define positives.
    z_a: (batch_size, d)
    z_b: (batch_size, d)
    cond_labels: (batch_size,) tensor of condition IDs WITHOUT THE SUBJECT_K CONDITION LABELS!!

    Supervised InfoNCE where positives share 'cond_labels' across modalities.

    Parameters
    ----------
    z_a, z_b : torch.Tensor, shape (B, D)
    cond_labels : torch.Tensor, shape (B,)
      Condition/category ids; positives share same id.
    temperature : float

    Returns
    -------
    torch.Tensor
      Scalar loss.
    """
    batch_size = z_a.size(0)

    # Cosine similarities
    sim_ab = torch.matmul(z_a, z_b.T) / temperature  # (N, N)
    sim_ba = torch.matmul(z_b, z_a.T) / temperature  # (N, N)

    # Create condition mask: mask[i, j] = 1 if same condition
    cond_mask = cond_labels.unsqueeze(0) == cond_labels.unsqueeze(1)  # (N, N)

    # Remove diagonal (self-pairs)
    cond_mask = cond_mask.fill_diagonal_(False)

    # ---- Loss for EEG -> fNIRS ----
    exp_sim = torch.exp(sim_ab)
    denom = exp_sim.sum(dim=1, keepdim=True)  # (N, 1)

    # numerator = sum of exp(sim) for positives (same condition)
    numerator = (exp_sim * cond_mask).sum(dim=1)

    # avoid zero positives (numerator=0)
    loss_ab = -torch.log((numerator + 1e-8) / (denom.squeeze(1) + 1e-8))

    # ---- Loss for fNIRS -> EEG ----
    exp_sim = torch.exp(sim_ba)
    denom = exp_sim.sum(dim=1, keepdim=True)
    numerator = (exp_sim * cond_mask).sum(dim=1)
    loss_ba = -torch.log((numerator + 1e-8) / (denom.squeeze(1) + 1e-8))

    # Average both directions
    loss = (loss_ab + loss_ba) / 2.0
    return loss.mean()


# prepare this for testing the model capabilities
def get_features(encoder_EEG, encoder_EEG_trans, dataloader, do_random_occlusion=False, device="cuda", channels=None, interp_option=None):
    """
    do here an evaluation of the similarities
    capabilities for prediction.

    Extract normalized embeddings for EEG and fNIRS (optionally with occlusions).

     Parameters
     ----------
     encoder_EEG, encoder_fNIRs : nn.Module
        Encoders returning (embedding, pre_flat) with input (B, 1, C, T).
     dataloader : DataLoader
       Yields (eeg, fnirs, label, subject, hbo).
     do_random_occlusion : bool, default=False
       If True, apply random erasing and compute similarity drops.
     device : str
     channels : list[str] or None
       Required when `interp_option` uses ROI-based erasing.
     interp_option : {0,1,2} or None
     0: patch erasing; 1: coarse ROI; 2: granular ROI.

     Returns
     -------
     tuple
     If 'do_random_occlusion':
       (Z_eeg, Z_fnirs, y, subj, hbo,
        eeg_occ, fnirs_occ, sims_softmax,
        w_eeg, w_fnirs, sim_drop_mean, sim_drop_sm_mean,
        sim_drop_std, sim_drop_sm_std, None, None)
     Else:
       (Z_eeg, Z_fnirs, y, subj, hbo,
        None, None, None, None, None, None, None, None, None,
        eeg_imgs, fnirs_imgs)
    """

    encoder_EEG.eval()
    encoder_EEG_trans.eval()

    all_features_eeg = []
    all_features_fnirs = []
    all_features_eeg_trans = []
    all_features_fnirs_trans = []
    all_labels = []
    all_labels_subject = []
    all_labels_hbo = []
    all_eeg_occ = []
    all_fnirs_occ = []
    all_eeg_occ_trans = []
    all_fnirs_occ_trans = []
    all_images_eeg = []
    all_images_fnirs = []
    all_images_eeg_trans = []
    all_images_fnirs_trans = []

    sims = []
    weights_eeg = []
    weights_fnirs = []
    weights_eeg_trans = []
    weights_fnirs_trans = []
    # get the cosine similarity here..
    measure = nn.CosineSimilarity(dim=-1)
    measure_class = nn.CosineSimilarity(dim=-1)

    # define here the random erasing
    if interp_option == 1:
        random_erase_eeg = RandomErasingOneROIs(p=1.0, min_frac=0.4, max_frac=0.8, channels=channels)
    elif interp_option == 0:
        random_erase_eeg = transforms.RandomErasing(p=1.0, scale=(0.4, 0.8), ratio=(0.1, 5))
    elif interp_option == 2:
        random_erase_eeg = RandomErasingOneROIsMoreGranular(p=1.0, min_frac=0.05, max_frac=0.2, channels=channels)

    random_erase_fnirs = transforms.RandomErasing(p=1.0, scale=(0.02, 0.7), ratio=(0.1, 5))

    with torch.no_grad():
        for eeg, fnirs, y, subj, hbo, eeg_trans, fnirs_trans in dataloader:
            eeg_features = eeg.cuda()
            # fnirs_features = fnirs.cuda()
            eeg_features_trans = eeg_trans.cuda()
            # fnirs_features_trans = fnirs.cuda()

            all_images_eeg.append(eeg_features)
            all_images_eeg_trans.append(eeg_features)
            # all_images_fnirs.append(fnirs_features)

            eeg_features_occluded = eeg_features.clone().detach()
            eeg_features_occluded_trans = eeg_features_trans.clone().detach()
            # fnirs_features_occluded = fnirs_features.clone().detach()

            # do the random erasing here for each image
            if do_random_occlusion is True:
                for index_batch in range(0, eeg_features.size(0)):
                    eeg_features_occluded[index_batch] = torch.squeeze(random_erase_eeg(eeg_features[index_batch].unsqueeze(0)))
                    eeg_features_occluded_trans[index_batch] = torch.squeeze(random_erase_eeg(eeg_features_trans[index_batch].unsqueeze(0)))
                    # fnirs_features_occluded[index_batch] = torch.squeeze(random_erase_fnirs(fnirs_features[index_batch].unsqueeze(0)))
                all_eeg_occ.append(eeg_features_occluded)
                all_eeg_occ_trans.append(eeg_features_occluded_trans)
                # all_fnirs_occ.append(fnirs_features_occluded)

            z_eeg, _ = encoder_EEG(eeg_features.unsqueeze(1))
            z_eeg_trans, _ = encoder_EEG_trans(eeg_features_trans.unsqueeze(1))

            if do_random_occlusion is True:
                z_eeg_occ, _ = encoder_EEG(eeg_features_occluded.unsqueeze(1))
                z_eeg_occ_trans, _ = encoder_EEG_trans(eeg_features_occluded_trans.unsqueeze(1))

                sims.append(measure(z_eeg_occ, z_eeg_occ_trans))
                weights_eeg.append(z_eeg_occ.norm(dim=-1))
                weights_eeg_trans.append(z_eeg_occ_trans.norm(dim=-1))

            z_eeg_norm = F.normalize(z_eeg, dim=1)
            z_eeg_norm_trans = F.normalize(z_eeg_trans, dim=1)

            all_features_eeg.append(z_eeg_norm)
            all_features_eeg_trans.append(z_eeg_norm_trans)
            # all_features_fnirs.append(z_fnirs_norm)

            all_labels.append(y.to(device))
            all_labels_subject.append(subj.to(device))
            all_labels_hbo.append(hbo.to(device))

    y_all = torch.cat(all_labels)
    subj_all = torch.cat(all_labels_subject)
    hbo_all = torch.cat(all_labels_hbo)

    # calculate here the similarity differences on pairs per batch
    if do_random_occlusion is True:
        sim_representation = torch.cat(sims, dim=0)
        baseline = measure(torch.cat(all_features_eeg), torch.cat(all_features_eeg_trans))
        sims = baseline - sim_representation  # the higher the drop, the better

        # do this for avoid variability in the measures...
        average_sims = sims
        sims = F.softmax(sims, dim=-1)
        sims = sims.cpu().numpy()
        logger.info(f"The sim difference is {np.mean(sims)} baseline is {baseline} and sim_rep is {sim_representation}")

        return (
            torch.cat(all_features_eeg),
            torch.cat(all_features_eeg_trans),
            y_all,
            subj_all,
            hbo_all,
            torch.cat(all_eeg_occ),
            torch.cat(all_eeg_occ_trans),
            sims,
            torch.cat(weights_eeg, dim=0).cpu().numpy(),
            torch.cat(weights_eeg_trans, dim=0).cpu().numpy(),
            np.mean(average_sims.cpu().numpy()),
            np.mean(sims),
            np.std(average_sims.cpu().numpy()),
            np.std(sims),
            None,
            None,
        )
    else:
        return torch.cat(all_features_eeg), torch.cat(all_features_eeg_trans), y_all, subj_all, hbo_all, None, None, None, None, None, None, None, None, None, torch.cat(all_images_eeg), torch.cat(all_images_eeg_trans)


# define the nearest neighbors clusters for the experiment and add them to the test
def calculate_nearest_subject_neighors(tsne_representation: Any, nearest_neighbors: int, subject_labels: Any, index_subj: int, hbo_labels: Any):
    """
      Compute nearest subject centroids in t-SNE space (balanced by HbO label).

      Parameters
      ----------
      umap_representation : np.ndarray, shape (N, 2) or (N, d) -> this nearest neighbors calculation is made with the UMAP representation***
      nearest_neighbors : int
        Number of subject IDs to select (balanced pos/neg HbO).
      subject_labels : np.ndarray, shape (N,)
        Subject ids 0..19.
      index_subj : int
        Held-out subject id.
      hbo_labels : np.ndarray, shape (N,)
         0/1 labels to stratify neighbors.
    Returns
    -------
    (list[int], float, float)
      (closest_subject_ids, mean_distance_to_selected, mean_pos_neg_centroid_distance) -> with the calculated distances and subject ids
    """
    centroids = {}
    difference = {}

    hbo_positive = np.unique(subject_labels[np.where(hbo_labels == 0)[0]])
    hbo_negative = np.unique(subject_labels[np.where(hbo_labels == 1)[0]])

    for label in range(0, 20):
        # Select data points belonging to the current label
        points_in_cluster = tsne_representation[subject_labels == label]
        # Calculate the mean of these points to get the centroid
        centroids[label] = np.mean(points_in_cluster, axis=0)

    # calculate the euclidian distances of the representation
    for label in range(0, 20):
        # label will NEVER be the same subject idx to compare centroids
        if label != index_subj:
            difference[label] = np.linalg.norm(centroids[label] - centroids[index_subj])

    # the lbl values will be always different than index_subj comming as the subject out!!
    filtered_pos = {lbl: dist for lbl, dist in difference.items() if lbl in hbo_positive}
    filtered_neg = {lbl: dist for lbl, dist in difference.items() if lbl in hbo_negative}

    # found the farthest neighbors from the tsne distribution **based on the reverse parameter true
    closest_labels_pos = sorted(filtered_pos, key=filtered_pos.get, reverse=False)[: round(nearest_neighbors / 2)]
    closest_labels_neg = sorted(filtered_neg, key=filtered_neg.get, reverse=False)[: round(nearest_neighbors / 2)]

    # use just the nearest neighbors here not the farest..
    # far_labels_pos = sorted(filtered_pos, key=filtered_pos.get, reverse=True)[:round(nearest_neighbors/2)]
    # far_labels_neg = sorted(filtered_neg, key=filtered_neg.get, reverse=True)[:round(nearest_neighbors/2)]
    distances_between = []
    for index_pos_p in range(0, len(closest_labels_pos)):
        for index_neg_p in range(0, len(closest_labels_neg)):
            distances_between.append(np.linalg.norm(centroids[int(closest_labels_pos[index_pos_p])] - centroids[int(closest_labels_neg[index_neg_p])]))

    dist_mean_between_clusters = np.mean(np.array(distances_between))

    # only get the nearest neighbors here not the farest ones!!
    closest_labels = list(set(closest_labels_pos + closest_labels_neg))  # + far_labels_pos + far_labels_neg))

    # get here the mean and max distances from two neares neighbors here..
    chosen_distances = [difference[lbl] for lbl in closest_labels]
    dist_mean = float(np.mean(chosen_distances)) if chosen_distances else 0.0
    dist_max = float(np.max(chosen_distances)) if chosen_distances else 0.0

    return closest_labels, dist_mean, dist_mean_between_clusters


# do a normalization for tsne representation
def normalize_tsne_representation(tsne_selected: Any):
    min_vals = tsne_selected.min(axis=0)
    max_vals = tsne_selected.max(axis=0)

    # Normalize to [0, 1]
    tsne_norm = (tsne_selected - min_vals) / (max_vals - min_vals + 1e-8)

    return tsne_norm


# set here the function definition for the CLIP
def CLIP_train(
    dataset_train, test_dataset, labels_Subject, num_epochs, temperature_ini, folder_name_save, folder_models, batch_size, embed_dim, info, info_fnirs, interp_option, learning_rate, subject_start, components_tsne, n_subs_val, k_value_self, k_value_class, umap_feat
):
    """
    Define here the CLIP architecture for training
    calculating the InfoNCE loss between the EEG and fNIRs
    embeddings
    """

    # simulate 80/20 using 5 fold crossval
    # only do it one time - just one epoch of this!! on the training split only..
    n_splits = 5  # -> do it for having 80% for train and 20% for val JUST for this!
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    column_names = ["epochs", "val_acc", "val_pr", "val_re", "val_f1", "test_acc", "test_pr", "test_re", "test_f1", "loss_train", "loss_test"]
    column_names_short = ["epochs", "acc_k1", "acc_k2", "acc_k4", "acc_k6", "acc_k8", "acc_k10", "acc_k12", "acc_k14", "acc_k16", "acc_k18", "acc_k20", "acc_vote", "was_dist_val", "temp"]
    column_names_short_svm = ["epochs", "acc_s0.01", "acc_s0.1", "acc_s0.5", "acc_s1.0", "acc_svote", "was_dist_val", "temp"]
    column_names_short_mlp = ["epochs", "acc_mlp1", "acc_mlp2", "acc_mlp3", "acc_mlp_vote", "was_dist_val", "temp"]
    column_names_metrics = ["epochs", "silhoutte", "MI", "Wasserstein Distance", "silhoutte_val", "MI_val", "Wasserstein Distance_val"]
    column_names_sims_differences = ["epochs", "sims_batch", "sims_batch_after_softmax", "sims_batch_std", "sims_batch_after_softmax_std"]

    loss_classifier = nn.CrossEntropyLoss()

    # pca = PCA(n_components=15)

    for fold_inner, (train_index, val_index) in enumerate(kf.split(dataset_train)):
        for subj in range(subject_start, 20):
            # define here the variables for each subject fold for avoid overwritting on the same subject table report

            # define here the cross-validation across subjects ********
            # define here the tsne mapping per subject..
            # use tsne JUST FOR VISUALIZATION. This can't work as an anchor for doing classification this just work as a tentative visualization tool..
            tsne_map = TSNE(n_components=components_tsne, perplexity=30, init="pca", random_state=42, n_jobs=-1, verbose=2)
            # use umap as feature set for training - always initialize the seed above to leave this working all the time in the same way!!!
            umap_map = umap.UMAP(n_neighbors=5, min_dist=0.1, n_components=umap_feat, random_state=42, verbose=True, metric="euclidean")

            # define here the dfs
            dfs_results_knn = pd.DataFrame(columns=column_names_short)
            dfs_results_svm = pd.DataFrame(columns=column_names_short_svm)
            dfs_results_mlp = pd.DataFrame(columns=column_names_short_mlp)
            df_metrics_cluster = pd.DataFrame(columns=column_names_metrics)

            # define here the dfs for val
            dfs_results_knn_val = pd.DataFrame(columns=column_names_short)
            dfs_results_svm_val = pd.DataFrame(columns=column_names_short_svm)
            dfs_results_mlp_val = pd.DataFrame(columns=column_names_short_mlp)
            df_metrics_cluster_val = pd.DataFrame(columns=column_names_metrics)

            df_sims = pd.DataFrame(columns=column_names_sims_differences)

            n_subs = n_subs_val
            # Define the column names
            column_sub_names = [f"sub_{i + 1}" for i in range(n_subs)]
            column_sub_names = column_sub_names + ["dist_mean", "dist_between"]
            # Create an empty DataFrame with the specified columns
            df_subs_sel = pd.DataFrame(columns=column_sub_names)

            time_vector = []
            silhoutte_vals = []
            mi_vals = []
            wass_dist_vals = []
            silhoutte_vals_val = []
            mi_vals_val = []
            wass_dist_vals_val = []

            # define the models here
            # define here the encoders per subject in this evaluation
            encoder_EEG = EEGNet(n_channels=43, embed_dim=embed_dim, n_times=548).cuda()  # ShallowConvNet_EEG(input_shape=[1, 43, 548], embed_dim=embed_dim, use_norm=False).cuda()
            encoder_EEG_trans = EEGNet(n_channels=43, embed_dim=embed_dim, n_times=548).cuda()
            # encoder_fNIRs = fNIRsNet(n_channels=76, embed_dim=embed_dim, n_times=9).cuda() #ShallowConvNet_fNIRs(input_shape=[1, 80, 9], embed_dim=embed_dim, use_norm=False).cuda()
            classifier = ReLUClassifier(input_dim=encoder_EEG.flatten_dim + encoder_EEG_trans.flatten_dim, hidden_dim=200, num_classes=20).cuda()  # define the classes for the 20 subjects

            # xavier initialization here
            encoder_EEG.apply(init_weights_xavier)
            encoder_EEG_trans.apply(init_weights_xavier)
            # encoder_fNIRs.apply(init_weights_xavier)

            # define the optimizer here..
            optimizer = torch.optim.AdamW(list(encoder_EEG.parameters()) + list(encoder_EEG_trans.parameters()), lr=learning_rate)
            scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
            # define the classifier optimizer here..
            optimizer_classifier = torch.optim.AdamW(list(classifier.parameters()), lr=learning_rate)
            scheduler_classifier = CosineAnnealingLR(optimizer_classifier, T_max=num_epochs, eta_min=5e-5)

            # get the generator for the train_loader here..
            gen_train = torch.Generator()
            gen_train.manual_seed(42)

            # Create subsets for training and validation based on the indices
            train_subset = Subset(dataset_train, train_index)
            val_subset = Subset(dataset_train, val_index)
            labels_subject_VAL = labels_Subject[val_index]

            # vary this factor what is the more desirable
            k_self = k_value_self  # get the k_value from the outer loop..
            k_class = k_value_class

            for epoch in range(0, num_epochs):
                # change the batchsize across the training epochs

                # leave the batch_size constant for not including non-linearities here..
                current_batch_size = batch_size  # for now use the same batch size
                # DO THIS LOGARITHMIC INCREASING FOR SMOOTHING THE CLUSTER CREATION and finding the sweet spot per subject!! PLOT THE LOGARITMIC INCREASING AT THE END

                factor_self = np.log(1 + k_self * (epoch + 1) / 400) / np.log(1 + k_self)
                # going from temperature_ini to 1 for self-supervised loss
                temperature_self = temperature_ini + (1 - temperature_ini) * factor_self

                factor_class = np.log(1 + k_class * (epoch + 1) / 400) / np.log(1 + k_class)
                # going from temperature_ini to 1 for supervised loss
                temperature_class = temperature_ini + (1 - temperature_ini) * factor_class

                logger.info(f"self-supervised batch_size is {current_batch_size}")
                logger.info(f"self-supervised temperature is {temperature_self}")

                # define the dataloaders here for the evaluation 80% for self-supervised, 20% for training (leaving subject out), and leaving test out for sure!!
                train_loader = DataLoader(train_subset, batch_size=current_batch_size, shuffle=True, pin_memory=True, num_workers=4, generator=gen_train, worker_init_fn=worker_init_fn)
                val_loader = DataLoader(val_subset, batch_size=current_batch_size, shuffle=False, pin_memory=True, num_workers=4, worker_init_fn=worker_init_fn)
                data_loader_test = DataLoader(test_dataset, batch_size=current_batch_size, shuffle=False, pin_memory=True, num_workers=4, worker_init_fn=worker_init_fn)

                # set here the model mode for training back
                # be sure to add this
                encoder_EEG.train()
                encoder_EEG_trans.train()

                embeddings_eeg = []
                embeddings_eeg_trans = []

                for eeg_imgs, fnirs_imgs, labels, subject_labels, labels_hbo, eeg_imgs_trans, fnirs_imgs_trans in train_loader:
                    # do contrastive learning first here..
                    eeg_features = eeg_imgs.cuda()
                    fnirs_features = fnirs_imgs.cuda()
                    eeg_features_trans = eeg_imgs_trans.cuda()
                    fnirs_features_trans = fnirs_imgs_trans.cuda()

                    z_eeg, eeg_flat = encoder_EEG(eeg_features.unsqueeze(1))
                    # z_fnirs, fnirs_flat = encoder_fNIRs(fnirs_features.unsqueeze(1))
                    z_eeg_trans, eeg_flat_trans = encoder_EEG_trans(eeg_features_trans.unsqueeze(1))

                    z_eeg_norm = F.normalize(z_eeg, dim=1)
                    z_eeg_trans_norm = F.normalize(z_eeg_trans, dim=1)
                    # joined the normalized features of the eeg_flat_transformation..
                    out_class = classifier(eeg_flat, eeg_flat_trans)

                    # append the embeddings
                    if z_eeg_norm.shape[0] == current_batch_size:
                        embeddings_eeg.append(z_eeg_norm.detach().cpu().numpy())
                        embeddings_eeg_trans.append(z_eeg_trans_norm.detach().cpu().numpy())

                    # remove the subejct info from the clip classification loss
                    info_nce_loss = clip_loss(z_eeg_norm, z_eeg_trans_norm, temperature=temperature_self)
                    info_nce_loss_class = clip_loss_conditional(z_eeg_norm[subject_labels != subj], z_eeg_trans_norm[subject_labels != subj], labels_hbo.cuda().long()[subject_labels != subj], temperature_class)

                    # this is just for following the paper presented by Chen et al 2022 - Apple paper!!
                    class_loss = loss_classifier(out_class, subject_labels.cuda().long())

                    # REWRITE THE LOSS HERE WITH BOTH COMPONENTS TOGETHER!!
                    loss_general = (info_nce_loss + info_nce_loss_class) / 2 + 0.01 * class_loss  # 0.15 * info_nce_loss + 0.85 * info_nce_loss_class  #+ 0.01 * class_loss

                    optimizer.zero_grad()
                    optimizer_classifier.zero_grad()
                    # info_nce_loss.backward()
                    loss_general.backward()
                    optimizer.step()
                    scheduler.step()
                    optimizer_classifier.step()
                    scheduler_classifier.step()

                del eeg_features, fnirs_features, eeg_features_trans, fnirs_features_trans, z_eeg, z_eeg_trans, z_eeg_norm, z_eeg_trans_norm

                # get this after each self-supervised training epoch # don't use it for now if this is not necessary ** just for eventual fine-tuning
                feature_extractor_EEG = EEGNetFeatureExtractor(n_channels=43)
                feature_extractor_EEG_trans = EEGNetFeatureExtractor(n_channels=43)
                # feature_extractor_fNIRs = fNIRsNetFeatureExtractor(n_channels=76)

                # Only load matching keys (exclude 'head')
                pretrained_dict_EEG = encoder_EEG.state_dict()
                feature_extractor_dict_EEG = feature_extractor_EEG.state_dict()
                pretrained_dict_EEG_trans = encoder_EEG_trans.state_dict()
                feature_extractor_dict_EEG_trans = feature_extractor_EEG_trans.state_dict()

                # Filter out head weights
                filtered_dict_EEG = {k: v for k, v in pretrained_dict_EEG.items() if k in feature_extractor_dict_EEG}
                feature_extractor_EEG.load_state_dict(filtered_dict_EEG)
                filtered_dict_EEG_trans = {k: v for k, v in pretrained_dict_EEG_trans.items() if k in feature_extractor_dict_EEG_trans}
                feature_extractor_EEG_trans.load_state_dict(filtered_dict_EEG_trans)

                # define here heads for infering the GradCAM in case
                att_head_EEG = AttHead(flatten_dim=feature_extractor_EEG.flatten_dim, embed_dim=embed_dim)
                att_head_EEG_trans = AttHead(flatten_dim=feature_extractor_EEG_trans.flatten_dim, embed_dim=embed_dim)

                # get the weights here from the whole encoders
                att_EEG_dict = att_head_EEG.state_dict()
                att_EEG_dict_trans = att_head_EEG_trans.state_dict()

                # Filter out head weights
                filtered_dict_head_EEG = {k: v for k, v in pretrained_dict_EEG.items() if k in att_EEG_dict}
                att_head_EEG.load_state_dict(filtered_dict_head_EEG)
                filtered_dict_head_EEG_trans = {k: v for k, v in pretrained_dict_EEG_trans.items() if k in att_EEG_dict_trans}
                att_head_EEG_trans.load_state_dict(filtered_dict_head_EEG_trans)

                logger.info(f"Epoch {epoch + 1}: Loss = {loss_general.item():.4f}, and subject {subj}")
                # get the evaluation accuracies based on logistic regression on the same dataset for now..
                feat_train_eeg, feat_train_eeg_trans, train_labels, train_sub_labels, train_hbo_labels, _, _, _, _, _, _, _, _, _, _, _ = get_features(
                    encoder_EEG=encoder_EEG, encoder_EEG_trans=encoder_EEG_trans, dataloader=train_loader, device="cuda", channels=info.ch_names, interp_option=interp_option
                )
                if interp_option != 3:
                    feat_val_eeg, feat_val_eeg_trans, val_labels, val_sub_labels, val_hbo_labels, val_eeg_occ, val_eeg_occ_trans, sims_val, weights_eeg_val, weights_eeg_val_trans, SIMS_val, SIMS_soft_val, SIM_std_val, SIM_soft_val_std, _, _ = get_features(
                        encoder_EEG=encoder_EEG, encoder_EEG_trans=encoder_EEG_trans, dataloader=val_loader, do_random_occlusion=True, device="cuda", channels=info.ch_names, interp_option=interp_option
                    )
                    feat_test_eeg, feat_test_eeg_trans, test_labels, test_sub_labels, test_hbo_labels, test_eeg_occ, test_eeg_occ_trans, sims_test, weights_eeg_test, weights_eeg_test_trans, SIMS_test, SIMS_soft_test, SIM_std_test, SIM_soft_test_std, _, _ = get_features(
                        encoder_EEG=encoder_EEG, encoder_EEG_trans=encoder_EEG_trans, dataloader=data_loader_test, do_random_occlusion=True, device="cuda", channels=info.ch_names, interp_option=interp_option
                    )
                else:
                    feat_val_eeg, feat_val_eeg_trans, val_labels, val_sub_labels, val_hbo_labels, _, _, _, _, _, _, _, _, _, val_eeg_imgs, val_eeg_imgs_trans = get_features(
                        encoder_EEG=encoder_EEG, encoder_EEG_trans=encoder_EEG_trans, dataloader=val_loader, do_random_occlusion=False, device="cuda", channels=info.ch_names, interp_option=None
                    )
                    feat_test_eeg, feat_test_eeg_trans, test_labels, test_sub_labels, test_hbo_labels, _, _, _, _, _, _, _, _, _, test_eeg_imgs, test_eeg_imgs_trans = get_features(
                        encoder_EEG=encoder_EEG, encoder_EEG_trans=encoder_EEG_trans, dataloader=data_loader_test, do_random_occlusion=False, device="cuda", channels=info.ch_names, interp_option=None
                    )

                if interp_option != 3:
                    # compare the sizes of the evaluation => DO THIS FOR EVALUATING INTERPRETABILITY..
                    assert sims_val.shape[0] == val_eeg_occ.shape[0] == val_eeg_occ_trans.shape[0]
                    EEG_xai = np.zeros((43, 548))  # create image with the size of EEG representation
                    EEG_xai_trans = np.zeros((43, 548))  # create image with the size of EEG representation
                    weights = list(zip(weights_eeg_val, weights_eeg_val_trans))

                    # do here the relevance calculation for occlussion analysis..
                    for n in range(val_eeg_occ.shape[0]):
                        eeg_2d = val_eeg_occ[n].unsqueeze(0).cpu().numpy().transpose((1, 2, 0)).sum(axis=-1)
                        eeg_2d_trans = val_eeg_occ_trans[n].unsqueeze(0).cpu().numpy().transpose((1, 2, 0)).sum(axis=-1)

                        joint_similarity = sims_val[n]
                        weight = weights[n]

                        # apply here the similarity in the zero regions...
                        if weight[0] < weight[1]:
                            EEG_xai[eeg_2d == 0] += joint_similarity
                        else:
                            EEG_xai_trans[eeg_2d_trans == 0] += joint_similarity

                else:  # do this for interaction-CAM ** modify grad-CAM if this is necessaary for further implementations..Take into account you need to change the code with the new variablles in case you need it after!!****
                    grad_cam = GradCAM(encoder_EEG=feature_extractor_EEG, encoder_fNIRs=feature_extractor_EEG_trans, attentional_head_eeg=att_head_EEG, attentional_head_fnirs=att_head_EEG_trans).cuda()
                    score = grad_cam(val_eeg_imgs.unsqueeze(1), val_eeg_imgs_trans.unsqueeze(1))

                    # check the projections here
                    grad_cam.zero_grad()
                    score.mean().backward()
                    G_EEG = grad_cam.gradients["EEG"]
                    G_fNIRs = grad_cam.gradients["fNIRs"]

                    # just do the attentional part for now.. This is the more stable version...
                    B, D, H, W = grad_cam.features["EEG"].size()
                    reshaped_EEG = grad_cam.features["EEG"].permute(0, 2, 3, 1).reshape(B, H * W, D)

                    B, D, H, W = grad_cam.features["fNIRs"].size()
                    reshaped_fNIRs = grad_cam.features["fNIRs"].permute(0, 2, 3, 1).reshape(B, H * W, D)

                    features_EEG_query, features_fNIRs_query = reshaped_EEG.mean(1).unsqueeze(1), reshaped_fNIRs.mean(1).unsqueeze(1)
                    attn_EEG = (features_EEG_query @ reshaped_EEG.transpose(-2, -1)).softmax(dim=-1)
                    attn_fNIRs = (features_fNIRs_query @ reshaped_fNIRs.transpose(-2, -1)).softmax(dim=-1)
                    att_reduced_EEG = (attn_EEG @ reshaped_EEG).squeeze(1)
                    att_reduced_fNIRs = (attn_fNIRs @ reshaped_fNIRs).squeeze(1)
                    joint_weight = att_reduced_EEG * att_reduced_fNIRs

                    joint_weight_EEG = joint_weight.unsqueeze(-1).unsqueeze(-1).expand_as(grad_cam.features["EEG"])
                    joint_weight_fNIRs = joint_weight.unsqueeze(-1).unsqueeze(-1).expand_as(grad_cam.features["fNIRs"])

                    feats_EEG = grad_cam.features["EEG"] * joint_weight_EEG
                    feats_fNIRs = grad_cam.features["fNIRs"] * joint_weight_fNIRs

                    feats_EEG = weight_activation(feats_EEG, G_EEG)
                    feats_fNIRs = weight_activation(feats_fNIRs, G_fNIRs)
                    # rezize the projections here
                    EEG_xai = np.mean(cv2.resize(feats_EEG, (548, 43)), axis=2)
                    EEG_xai_trans = np.mean(cv2.resize(feats_fNIRs, (9, 76)), axis=2)

                # do the interpretability estimation here...
                EEG_xai = (EEG_xai - np.min(EEG_xai)) / (np.max(EEG_xai) - np.min(EEG_xai))
                # fNIRs_xai = (fNIRs_xai - np.min(fNIRs_xai)) / (np.max(fNIRs_xai) - np.min(fNIRs_xai))
                EEG_xai_trans = (EEG_xai_trans - np.min(EEG_xai_trans)) / (np.max(EEG_xai_trans) - np.min(EEG_xai_trans))

                # plot here the topo-plot for the relevances across the epocs in the val dataset..
                plotting_relevance_map_eeg_fnirs(eeg_xai_channels=EEG_xai, info=info, epoch=epoch, folder_imgs=folder_name_save)

                # plot the EEG image here
                fig, ax = plt.subplots(figsize=(10, 10))
                im = ax.imshow(EEG_xai, cmap="jet")
                plt.title(f"EEG relevance epoch {epoch}")
                plt.savefig(f"{folder_name_save}/relevance_map_{subj}")
                plt.close("all")

                # do here the downstream task evaluations
                if (epoch + 1) % 10 == 0 or epoch + 1 == 1:
                    # DO THE TSNET FIT TRANSFORM HERE!! make this for sharing the same embedding space in tsne!!! ADD ALL THE TRIALS IN THE SAME MAPPING!!
                    features_tsne_ALL = tsne_map.fit_transform(
                        np.concatenate(
                            (torch.cat((feat_train_eeg, feat_train_eeg_trans), dim=1).detach().cpu().numpy(), torch.cat((feat_val_eeg, feat_val_eeg_trans), dim=1).detach().cpu().numpy(), torch.cat((feat_test_eeg, feat_test_eeg_trans), dim=1).detach().cpu().numpy()),
                            axis=0,
                        )
                    )
                    # DO THE UMAP HERE FOR FEATURE EXTRACTION AND CLASSIFICATION!!
                    features_umap_ALL = umap_map.fit_transform(
                        np.concatenate(
                            (torch.cat((feat_train_eeg, feat_train_eeg_trans), dim=1).detach().cpu().numpy(), torch.cat((feat_val_eeg, feat_val_eeg_trans), dim=1).detach().cpu().numpy(), torch.cat((feat_test_eeg, feat_test_eeg_trans), dim=1).detach().cpu().numpy()),
                            axis=0,
                        )
                    )

                    # GET THE T-SNE FEATURES HERE!!
                    features_tsne_train = features_tsne_ALL[0 : feat_train_eeg.shape[0], :]
                    features_tsne_val = features_tsne_ALL[feat_train_eeg.shape[0] : feat_train_eeg.shape[0] + feat_val_eeg.shape[0], :]
                    features_tsne_test = features_tsne_ALL[feat_train_eeg.shape[0] + feat_val_eeg.shape[0] : feat_train_eeg.shape[0] + feat_val_eeg.shape[0] + feat_test_eeg.shape[0], :]

                    # GET THE UMAP FEATURES HERE!!
                    features_UMAP_train = features_umap_ALL[0 : feat_train_eeg.shape[0], :]
                    features_UMAP_val = features_umap_ALL[feat_train_eeg.shape[0] : feat_train_eeg.shape[0] + feat_val_eeg.shape[0], :]
                    features_UMAP_test = features_umap_ALL[feat_train_eeg.shape[0] + feat_val_eeg.shape[0] : feat_train_eeg.shape[0] + feat_val_eeg.shape[0] + feat_test_eeg.shape[0], :]

                    # plot the tsne features here and save that in the corresponding folder. This plot is related to the stimuli type (i.e Pitch v.s No-Pitch) - tsne
                    plot_tsne_feat(tsne_feat=features_tsne_train, labels=train_labels.detach().cpu().numpy(), title=f"Train t-sne Epoch {epoch + 1}", epoch=str(epoch + 1), suffix="train", folder_name=folder_name_save, class_value="stimulus type", subj=subj)
                    plot_tsne_feat(tsne_feat=features_tsne_val, labels=val_labels.detach().cpu().numpy(), title=f"Val t-sne Epoch {epoch + 1}", epoch=str(epoch + 1), suffix="val", folder_name=folder_name_save, class_value="stimulus type", subj=subj)
                    plot_tsne_feat(tsne_feat=features_tsne_test, labels=test_labels.detach().cpu().numpy(), title=f"test t-sne Epoch {epoch + 1}", epoch=str(epoch + 1), suffix="test", folder_name=folder_name_save, class_value="stimulus type", subj=subj)

                    # plot here the HbO positive and HbO negative labels in the tsne representation - tsne
                    plot_tsne_feat(
                        tsne_feat=features_tsne_train, labels=train_hbo_labels.detach().cpu().numpy(), title=f"Train t-sne Epoch {epoch + 1}", epoch=str(epoch + 1), suffix="train_hbo", folder_name=folder_name_save, class_value="HbO positive/negative", subj=subj
                    )
                    plot_tsne_feat(tsne_feat=features_tsne_val, labels=val_hbo_labels.detach().cpu().numpy(), title=f"Val t-sne Epoch {epoch + 1}", epoch=str(epoch + 1), suffix="val_hbo", folder_name=folder_name_save, class_value="HbO positive/negative", subj=subj)
                    plot_tsne_feat(tsne_feat=features_tsne_test, labels=test_hbo_labels.detach().cpu().numpy(), title=f"test t-sne Epoch {epoch + 1}", epoch=str(epoch + 1), suffix="test_hbo", folder_name=folder_name_save, class_value="HbO positive/negative", subj=subj)

                    # plot the tsne features here and save that in the corresponding folder. This plot is related to the stimuli type (i.e Pitch v.s No-Pitch) - umap
                    plot_tsne_feat(tsne_feat=features_UMAP_train, labels=train_labels.detach().cpu().numpy(), title=f"Train umap Epoch {epoch + 1}", epoch=str(epoch + 1), suffix="train_umap", folder_name=folder_name_save, class_value="stimulus type", subj=subj)
                    plot_tsne_feat(tsne_feat=features_UMAP_val, labels=val_labels.detach().cpu().numpy(), title=f"Val t-sne umap {epoch + 1}", epoch=str(epoch + 1), suffix="val_umap", folder_name=folder_name_save, class_value="stimulus type", subj=subj)
                    plot_tsne_feat(tsne_feat=features_UMAP_test, labels=test_labels.detach().cpu().numpy(), title=f"test umap Epoch {epoch + 1}", epoch=str(epoch + 1), suffix="test_umap", folder_name=folder_name_save, class_value="stimulus type", subj=subj)

                    # plot here the HbO positive and HbO negative labels in the tsne representation - umap
                    plot_tsne_feat(
                        tsne_feat=features_UMAP_train, labels=train_hbo_labels.detach().cpu().numpy(), title=f"Train umap Epoch {epoch + 1}", epoch=str(epoch + 1), suffix="train_hbo_umap", folder_name=folder_name_save, class_value="HbO positive/negative", subj=subj
                    )
                    plot_tsne_feat(tsne_feat=features_UMAP_val, labels=val_hbo_labels.detach().cpu().numpy(), title=f"Val umap Epoch {epoch + 1}", epoch=str(epoch + 1), suffix="val_hbo_umap", folder_name=folder_name_save, class_value="HbO positive/negative", subj=subj)
                    plot_tsne_feat(
                        tsne_feat=features_UMAP_test, labels=test_hbo_labels.detach().cpu().numpy(), title=f"test umap Epoch {epoch + 1}", epoch=str(epoch + 1), suffix="test_hbo_umap", folder_name=folder_name_save, class_value="HbO positive/negative", subj=subj
                    )

                    # calculate the metrics here depending on the associated labels..
                    sil_score = silhouette_score(features_tsne_train, train_sub_labels.detach().cpu().numpy())
                    kmeans = KMeans(n_clusters=len(set(train_sub_labels.detach().cpu().numpy())), n_init=10).fit(features_tsne_train)
                    mi = normalized_mutual_info_score(kmeans.labels_, train_sub_labels.detach().cpu().numpy())

                    sil_score_val = silhouette_score(features_tsne_val, val_sub_labels.detach().cpu().numpy())
                    kmeans_val = KMeans(n_clusters=len(set(val_sub_labels.detach().cpu().numpy())), n_init=10).fit(features_tsne_val)
                    mi_val = normalized_mutual_info_score(kmeans_val.labels_, val_sub_labels.detach().cpu().numpy())

                    # This plot is related to the subject_id subjects_id (1-20)
                    plot_tsne_feat(
                        tsne_feat=features_tsne_train,
                        labels=train_sub_labels.detach().cpu().numpy(),
                        title=f"Train t-sne Epoch {epoch + 1}, silhoutte={sil_score}, MI={mi}",
                        epoch=str(epoch + 1),
                        suffix="train_subject",
                        folder_name=folder_name_save,
                        class_value="subject",
                        subj=subj,
                    )
                    plot_tsne_feat(
                        tsne_feat=features_tsne_val,
                        labels=val_sub_labels.detach().cpu().numpy(),
                        title=f"Val t-sne Epoch {epoch + 1}, silhoutte={sil_score_val}, MI={mi_val}",
                        epoch=str(epoch + 1),
                        suffix="val_subject",
                        folder_name=folder_name_save,
                        class_value="subject",
                        subj=subj,
                    )
                    plot_tsne_feat(tsne_feat=features_tsne_test, labels=test_sub_labels.detach().cpu().numpy(), title=f"Test t-sne Epoch {epoch + 1}", epoch=str(epoch + 1), suffix="test_subject", folder_name=folder_name_save, class_value="subject", subj=subj)

                    plot_tsne_feat(
                        tsne_feat=features_UMAP_train,
                        labels=train_sub_labels.detach().cpu().numpy(),
                        title=f"Train umap Epoch {epoch + 1}, silhoutte={sil_score}, MI={mi}",
                        epoch=str(epoch + 1),
                        suffix="train_subject_umap",
                        folder_name=folder_name_save,
                        class_value="subject",
                        subj=subj,
                    )
                    plot_tsne_feat(
                        tsne_feat=features_UMAP_val,
                        labels=val_sub_labels.detach().cpu().numpy(),
                        title=f"Val umap Epoch {epoch + 1}, silhoutte={sil_score_val}, MI={mi_val}",
                        epoch=str(epoch + 1),
                        suffix="val_subject_umap",
                        folder_name=folder_name_save,
                        class_value="subject",
                        subj=subj,
                    )
                    plot_tsne_feat(tsne_feat=features_UMAP_test, labels=test_sub_labels.detach().cpu().numpy(), title=f"Test umap Epoch {epoch + 1}", epoch=str(epoch + 1), suffix="test_subject_umap", folder_name=folder_name_save, class_value="subject", subj=subj)

                    logger.info(f"The observed silhoutte score for epoch {epoch} is {sil_score} and MI is {mi}, and for validation is {sil_score_val} and MI is {mi_val}")

                    # do the analysis on the time vector here..
                    time_vector.append(epoch + 1)
                    silhoutte_vals.append(sil_score)
                    mi_vals.append(mi)
                    silhoutte_vals_val.append(sil_score_val)
                    mi_vals_val.append(mi_val)

                    # calculate here the wasserstein distance
                    wd = wasserstein_distance(
                        np.mean(np.reshape(np.array(embeddings_eeg), (np.array(embeddings_eeg).shape[0] * current_batch_size, embed_dim)), axis=1),
                        np.mean(np.reshape(np.array(embeddings_eeg_trans), (np.array(embeddings_eeg_trans).shape[0] * current_batch_size, embed_dim)), axis=1),
                    )
                    wass_dist_vals.append(wd)

                    wd_val = wasserstein_distance(np.mean(feat_val_eeg.detach().cpu().numpy(), axis=1), np.mean(feat_val_eeg_trans.detach().cpu().numpy(), axis=1))
                    wass_dist_vals_val.append(wd_val)

                    # plotting the silhoutte and normalized_mutual_information
                    plotting_twinx_variables(time_vector=np.array(time_vector), data1=np.array(silhoutte_vals), data2=np.array(mi_vals), title="Silhoutte & MI", x_label="Epochs", y_label1="silhoutte", y_label2="MI", folder_images=folder_name_save, subj=subj)
                    plotting_twinx_variables(
                        time_vector=np.array(time_vector), data1=np.array(wass_dist_vals), data2=np.array(mi_vals), title="Wasserstein Distance & MI", x_label="Epochs", y_label1="Wasserstein_Distance", y_label2="MI", folder_images=folder_name_save, subj=subj
                    )

                    plotting_twinx_variables(
                        time_vector=np.array(time_vector), data1=np.array(silhoutte_vals_val), data2=np.array(mi_vals_val), title="Silhoutte & MI", x_label="Epochs", y_label1="silhoutte_val", y_label2="MI_val", folder_images=folder_name_save, subj=subj
                    )
                    plotting_twinx_variables(
                        time_vector=np.array(time_vector),
                        data1=np.array(wass_dist_vals_val),
                        data2=np.array(mi_vals_val),
                        title="Wasserstein Distance & MI",
                        x_label="Epochs",
                        y_label1="Wasserstein_Distance_val",
                        y_label2="MI_val",
                        folder_images=folder_name_save,
                        subj=subj,
                    )

                    if interp_option != 2:
                        df_sims.loc[len(df_sims)] = [epoch + 1, SIMS_val, SIMS_soft_val, SIM_std_val, SIM_soft_val_std]
                        df_sims.to_csv(f"{folder_models}/results_sims_subj_{subj}.csv", index=False)

                    df_metrics_cluster.loc[len(df_metrics_cluster)] = [epoch + 1, sil_score, mi, wd, sil_score_val, mi_val, wd_val]
                    df_metrics_cluster.to_csv(f"{folder_models}/results_metrics_subj_{subj}.csv", index=False)

                    # compute the nearest neighbors and distances here..
                    nearest_subject_indeces, dist_mean, dist_max = calculate_nearest_subject_neighors(
                        tsne_representation=features_UMAP_train[:, :], nearest_neighbors=n_subs, subject_labels=train_sub_labels.detach().cpu().numpy(), index_subj=subj, hbo_labels=train_hbo_labels.detach().cpu().numpy()
                    )

                    logger.info(f"The nearest neighbors of this evaluation are {nearest_subject_indeces}")

                    df_subs_sel.loc[len(df_subs_sel)] = np.concatenate((nearest_subject_indeces, [dist_mean, dist_max]))
                    df_subs_sel.to_csv(f"{folder_models}/subjects_selected_{subj}.csv", index=False)

                    # define and run the classifier here - uncomment this just reference of initial test
                    # leave the subject **subj** information out from the validation set to do the finetuning with the information and labels from the subject subj
                    labels_val_sel = np.where((train_sub_labels.detach().cpu().numpy() != subj) & (np.isin(train_sub_labels.detach().cpu().numpy(), nearest_subject_indeces)))[0]
                    # for this leave the subject features inside NO LABEL is used from the subject id **subj**
                    labels_test_sel = np.where((test_sub_labels.detach().cpu().numpy() == subj) | (np.isin(test_sub_labels.detach().cpu().numpy(), nearest_subject_indeces)))[0]
                    labels_VAL_sel = np.where((val_sub_labels.detach().cpu().numpy() == subj) | (np.isin(val_sub_labels.detach().cpu().numpy(), nearest_subject_indeces)))[0]

                    # get the features here for train
                    feat_val_sel = features_UMAP_train[labels_val_sel, :]
                    labels_val_use = train_hbo_labels.detach().cpu().numpy()[labels_val_sel]

                    # get the features here for val and monitor them..DON'T USE THE ACCURACY JUST TO VISUALIZATION!!
                    feat_VAL_sel = features_UMAP_val[labels_VAL_sel, :]
                    labels_VAL_use = val_hbo_labels.detach().cpu().numpy()[labels_VAL_sel]
                    labels_VAL_sub_use = val_sub_labels.detach().cpu().numpy()[labels_VAL_sel]

                    # get here the features for test
                    feat_test_sel = features_UMAP_test[labels_test_sel, :]
                    labels_test_use = test_hbo_labels.detach().cpu().numpy()[labels_test_sel]
                    labels_test_sub_use = test_sub_labels.detach().cpu().numpy()[labels_test_sel]

                    # apply the MLP 50 here
                    classifier_mlp_hbo_1 = MLPClassifier(hidden_layer_sizes=(50), activation="relu", solver="adam", max_iter=1000, random_state=42, verbose=2)
                    classifier_mlp_hbo_1.fit(feat_val_sel, labels_val_use)
                    predictions_mlp_hbo_1_val = classifier_mlp_hbo_1.predict(feat_VAL_sel)
                    acc_mlp_hbo_sub_1_val = accuracy_score(labels_VAL_use[labels_VAL_sub_use == subj], predictions_mlp_hbo_1_val[labels_VAL_sub_use == subj])
                    predictions_mlp_hbo_1 = classifier_mlp_hbo_1.predict(feat_test_sel)
                    acc_mlp_hbo_sub_1 = accuracy_score(labels_test_use[labels_test_sub_use == subj], predictions_mlp_hbo_1[labels_test_sub_use == subj])

                    # apply the MLP 100 here
                    classifier_mlp_hbo_2 = MLPClassifier(hidden_layer_sizes=(100), activation="relu", solver="adam", max_iter=1000, random_state=42, verbose=2)
                    classifier_mlp_hbo_2.fit(feat_val_sel, labels_val_use)
                    predictions_mlp_hbo_2_val = classifier_mlp_hbo_2.predict(feat_VAL_sel)
                    acc_mlp_hbo_sub_2_val = accuracy_score(labels_VAL_use[labels_VAL_sub_use == subj], predictions_mlp_hbo_2_val[labels_VAL_sub_use == subj])
                    predictions_mlp_hbo_2 = classifier_mlp_hbo_2.predict(feat_test_sel)
                    acc_mlp_hbo_sub_2 = accuracy_score(labels_test_use[labels_test_sub_use == subj], predictions_mlp_hbo_2[labels_test_sub_use == subj])

                    # apply the MLP 200 here
                    classifier_mlp_hbo_3 = MLPClassifier(hidden_layer_sizes=(300), activation="relu", solver="adam", max_iter=1000, random_state=42, verbose=2)
                    classifier_mlp_hbo_3.fit(feat_val_sel, labels_val_use)
                    predictions_mlp_hbo_3_val = classifier_mlp_hbo_3.predict(feat_VAL_sel)
                    acc_mlp_hbo_sub_3_val = accuracy_score(labels_VAL_use[labels_VAL_sub_use == subj], predictions_mlp_hbo_3_val[labels_VAL_sub_use == subj])
                    predictions_mlp_hbo_3 = classifier_mlp_hbo_3.predict(feat_test_sel)
                    acc_mlp_hbo_sub_3 = accuracy_score(labels_test_use[labels_test_sub_use == subj], predictions_mlp_hbo_3[labels_test_sub_use == subj])

                    # apply majority voting here for MLP
                    class_voting_mlp = VotingClassifier(
                        estimators=[
                            ("m1", MLPClassifier(hidden_layer_sizes=(50), activation="relu", solver="adam", max_iter=1000)),
                            ("m2", MLPClassifier(hidden_layer_sizes=(100), activation="relu", solver="adam", max_iter=1000)),
                            ("m3", MLPClassifier(hidden_layer_sizes=(200), activation="relu", solver="adam", max_iter=1000)),
                        ],
                        voting="hard",
                    )
                    class_voting_mlp.fit(feat_val_sel, labels_val_use)
                    predictions_mlp_hbo_vote_val = class_voting_mlp.predict(feat_VAL_sel)
                    acc_mlp_vote_val = accuracy_score(labels_VAL_use[labels_VAL_sub_use == subj], predictions_mlp_hbo_vote_val[labels_VAL_sub_use == subj])
                    predictions_mlp_hbo_vote = class_voting_mlp.predict(feat_test_sel)
                    acc_mlp_vote = accuracy_score(labels_test_use[labels_test_sub_use == subj], predictions_mlp_hbo_vote[labels_test_sub_use == subj])

                    # append the df values here
                    dfs_results_mlp_val.loc[len(dfs_results_mlp_val)] = [epoch + 1, acc_mlp_hbo_sub_1_val, acc_mlp_hbo_sub_2_val, acc_mlp_hbo_sub_3_val, acc_mlp_vote_val, wd_val, temperature]
                    # save the results interim
                    dfs_results_mlp_val.to_csv(f"{folder_models}/results_mlp_subject_{subj}_val.csv", index=False)

                    # append the df values here
                    dfs_results_mlp.loc[len(dfs_results_mlp)] = [epoch + 1, acc_mlp_hbo_sub_1, acc_mlp_hbo_sub_2, acc_mlp_hbo_sub_3, acc_mlp_vote, wd_val, temperature_self]
                    # save the results interim
                    dfs_results_mlp.to_csv(f"{folder_models}/results_mlp_subject_{subj}.csv", index=False)

                    logger.info(f"The results for hbo positive/negative val epoch {epoch + 1} mlp are: Acc_mlp={acc_mlp_hbo_sub_1_val}, Acc_mlp2={acc_mlp_hbo_sub_2_val}, Acc_mlp3={acc_mlp_hbo_sub_1_val}, Acc_vote={acc_mlp_vote_val}")
                    logger.info(f"The results for hbo positive/negative test epoch {epoch + 1} mlp are: Acc_mlp={acc_mlp_hbo_sub_1}, Acc_mlp2={acc_mlp_hbo_sub_2}, Acc_mlp3={acc_mlp_hbo_sub_1}, Acc_vote={acc_mlp_vote}")

                    # apply knn 1 here
                    classifier_knn_hbo_1 = KNeighborsClassifier(n_neighbors=1, weights="distance")
                    classifier_knn_hbo_1.fit(feat_val_sel, labels_val_use)
                    predictions_knn_hbo_1_val = classifier_knn_hbo_1.predict(feat_VAL_sel)
                    acc_knn_hbo_sub_1_val = accuracy_score(labels_VAL_use[labels_VAL_sub_use == subj], predictions_knn_hbo_1_val[labels_VAL_sub_use == subj])
                    predictions_knn_hbo_1 = classifier_knn_hbo_1.predict(feat_test_sel)
                    acc_knn_hbo_sub_1 = accuracy_score(labels_test_use[labels_test_sub_use == subj], predictions_knn_hbo_1[labels_test_sub_use == subj])

                    # apply knn 2 here
                    classifier_knn_hbo_2 = KNeighborsClassifier(n_neighbors=2, weights="distance")
                    classifier_knn_hbo_2.fit(feat_val_sel, labels_val_use)
                    predictions_knn_hbo_2_val = classifier_knn_hbo_2.predict(feat_VAL_sel)
                    acc_knn_hbo_sub_2_val = accuracy_score(labels_VAL_use[labels_VAL_sub_use == subj], predictions_knn_hbo_2_val[labels_VAL_sub_use == subj])
                    predictions_knn_hbo_2 = classifier_knn_hbo_2.predict(feat_test_sel)
                    acc_knn_hbo_sub_2 = accuracy_score(labels_test_use[labels_test_sub_use == subj], predictions_knn_hbo_2[labels_test_sub_use == subj])

                    # apply knn 4 here
                    classifier_knn_hbo_4 = KNeighborsClassifier(n_neighbors=4, weights="distance")
                    classifier_knn_hbo_4.fit(feat_val_sel, labels_val_use)
                    predictions_knn_hbo_4_val = classifier_knn_hbo_4.predict(feat_VAL_sel)
                    acc_knn_hbo_sub_4_val = accuracy_score(labels_VAL_use[labels_VAL_sub_use == subj], predictions_knn_hbo_4_val[labels_VAL_sub_use == subj])
                    predictions_knn_hbo_4 = classifier_knn_hbo_4.predict(feat_test_sel)
                    acc_knn_hbo_sub_4 = accuracy_score(labels_test_use[labels_test_sub_use == subj], predictions_knn_hbo_4[labels_test_sub_use == subj])

                    # apply knn 6 here
                    classifier_knn_hbo_6 = KNeighborsClassifier(n_neighbors=6, weights="distance")
                    classifier_knn_hbo_6.fit(feat_val_sel, labels_val_use)
                    predictions_knn_hbo_6_val = classifier_knn_hbo_6.predict(feat_VAL_sel)
                    acc_knn_hbo_sub_6_val = accuracy_score(labels_VAL_use[labels_VAL_sub_use == subj], predictions_knn_hbo_6_val[labels_VAL_sub_use == subj])
                    predictions_knn_hbo_6 = classifier_knn_hbo_6.predict(feat_test_sel)
                    acc_knn_hbo_sub_6 = accuracy_score(labels_test_use[labels_test_sub_use == subj], predictions_knn_hbo_6[labels_test_sub_use == subj])

                    # apply knn 8 here
                    classifier_knn_hbo_8 = KNeighborsClassifier(n_neighbors=8, weights="distance")
                    classifier_knn_hbo_8.fit(feat_val_sel, labels_val_use)
                    predictions_knn_hbo_8_val = classifier_knn_hbo_8.predict(feat_VAL_sel)
                    acc_knn_hbo_sub_8_val = accuracy_score(labels_VAL_use[labels_VAL_sub_use == subj], predictions_knn_hbo_8_val[labels_VAL_sub_use == subj])
                    predictions_knn_hbo_8 = classifier_knn_hbo_8.predict(feat_test_sel)
                    acc_knn_hbo_sub_8 = accuracy_score(labels_test_use[labels_test_sub_use == subj], predictions_knn_hbo_8[labels_test_sub_use == subj])

                    # apply knn 10 here
                    classifier_knn_hbo_10 = KNeighborsClassifier(n_neighbors=10, weights="distance")
                    classifier_knn_hbo_10.fit(feat_val_sel, labels_val_use)
                    predictions_knn_hbo_10_val = classifier_knn_hbo_1.predict(feat_VAL_sel)
                    acc_knn_hbo_sub_10_val = accuracy_score(labels_VAL_use[labels_VAL_sub_use == subj], predictions_knn_hbo_10_val[labels_VAL_sub_use == subj])
                    predictions_knn_hbo_10 = classifier_knn_hbo_10.predict(feat_test_sel)
                    acc_knn_hbo_sub_10 = accuracy_score(labels_test_use[labels_test_sub_use == subj], predictions_knn_hbo_10[labels_test_sub_use == subj])

                    # apply knn 12 here
                    classifier_knn_hbo_12 = KNeighborsClassifier(n_neighbors=12, weights="distance")
                    classifier_knn_hbo_12.fit(feat_val_sel, labels_val_use)
                    predictions_knn_hbo_12_val = classifier_knn_hbo_12.predict(feat_VAL_sel)
                    acc_knn_hbo_sub_12_val = accuracy_score(labels_VAL_use[labels_VAL_sub_use == subj], predictions_knn_hbo_12_val[labels_VAL_sub_use == subj])
                    predictions_knn_hbo_12 = classifier_knn_hbo_12.predict(feat_test_sel)
                    acc_knn_hbo_sub_12 = accuracy_score(labels_test_use[labels_test_sub_use == subj], predictions_knn_hbo_12[labels_test_sub_use == subj])

                    # apply knn 14 here
                    classifier_knn_hbo_14 = KNeighborsClassifier(n_neighbors=14, weights="distance")
                    classifier_knn_hbo_14.fit(feat_val_sel, labels_val_use)
                    predictions_knn_hbo_14_val = classifier_knn_hbo_14.predict(feat_VAL_sel)
                    acc_knn_hbo_sub_14_val = accuracy_score(labels_VAL_use[labels_VAL_sub_use == subj], predictions_knn_hbo_14_val[labels_VAL_sub_use == subj])
                    predictions_knn_hbo_14 = classifier_knn_hbo_14.predict(feat_test_sel)
                    acc_knn_hbo_sub_14 = accuracy_score(labels_test_use[labels_test_sub_use == subj], predictions_knn_hbo_14[labels_test_sub_use == subj])

                    # apply knn 16 here
                    classifier_knn_hbo_16 = KNeighborsClassifier(n_neighbors=16, weights="distance")
                    classifier_knn_hbo_16.fit(feat_val_sel, labels_val_use)
                    predictions_knn_hbo_16_val = classifier_knn_hbo_16.predict(feat_VAL_sel)
                    acc_knn_hbo_sub_16_val = accuracy_score(labels_VAL_use[labels_VAL_sub_use == subj], predictions_knn_hbo_16_val[labels_VAL_sub_use == subj])
                    predictions_knn_hbo_16 = classifier_knn_hbo_16.predict(feat_test_sel)
                    acc_knn_hbo_sub_16 = accuracy_score(labels_test_use[labels_test_sub_use == subj], predictions_knn_hbo_16[labels_test_sub_use == subj])

                    # apply knn 18 here
                    classifier_knn_hbo_18 = KNeighborsClassifier(n_neighbors=18, weights="distance")
                    classifier_knn_hbo_18.fit(feat_val_sel, labels_val_use)
                    predictions_knn_hbo_18_val = classifier_knn_hbo_18.predict(feat_VAL_sel)
                    acc_knn_hbo_sub_18_val = accuracy_score(labels_VAL_use[labels_VAL_sub_use == subj], predictions_knn_hbo_18_val[labels_VAL_sub_use == subj])
                    predictions_knn_hbo_18 = classifier_knn_hbo_18.predict(feat_test_sel)
                    acc_knn_hbo_sub_18 = accuracy_score(labels_test_use[labels_test_sub_use == subj], predictions_knn_hbo_18[labels_test_sub_use == subj])

                    # apply knn 20 here
                    classifier_knn_hbo_20 = KNeighborsClassifier(n_neighbors=20, weights="distance")
                    classifier_knn_hbo_20.fit(feat_val_sel, labels_val_use)
                    predictions_knn_hbo_20_val = classifier_knn_hbo_20.predict(feat_VAL_sel)
                    acc_knn_hbo_sub_20_val = accuracy_score(labels_VAL_use[labels_VAL_sub_use == subj], predictions_knn_hbo_20_val[labels_VAL_sub_use == subj])
                    predictions_knn_hbo_20 = classifier_knn_hbo_20.predict(feat_test_sel)
                    acc_knn_hbo_sub_20 = accuracy_score(labels_test_use[labels_test_sub_use == subj], predictions_knn_hbo_20[labels_test_sub_use == subj])

                    # apply majority voting here
                    class_voting = VotingClassifier(
                        estimators=[
                            ("k2", KNeighborsClassifier(n_neighbors=2, weights="distance")),
                            ("k4", KNeighborsClassifier(n_neighbors=4, weights="distance")),
                            ("k6", KNeighborsClassifier(n_neighbors=6, weights="distance")),
                            ("k8", KNeighborsClassifier(n_neighbors=8, weights="distance")),
                            ("k10", KNeighborsClassifier(n_neighbors=10, weights="distance")),
                            ("k12", KNeighborsClassifier(n_neighbors=12, weights="distance")),
                            ("k14", KNeighborsClassifier(n_neighbors=14, weights="distance")),
                            ("k16", KNeighborsClassifier(n_neighbors=16, weights="distance")),
                            ("k18", KNeighborsClassifier(n_neighbors=18, weights="distance")),
                            ("k20", KNeighborsClassifier(n_neighbors=20, weights="distance")),
                        ],
                        voting="hard",
                    )
                    class_voting.fit(feat_val_sel, labels_val_use)
                    predictions_knn_hbo_vote_val = class_voting.predict(feat_VAL_sel)
                    acc_knn_vote_val = accuracy_score(labels_VAL_use[labels_VAL_sub_use == subj], predictions_knn_hbo_vote_val[labels_VAL_sub_use == subj])
                    predictions_knn_hbo_vote = class_voting.predict(feat_test_sel)
                    acc_knn_vote = accuracy_score(labels_test_use[labels_test_sub_use == subj], predictions_knn_hbo_vote[labels_test_sub_use == subj])

                    # append the df values here
                    dfs_results_knn_val.loc[len(dfs_results_knn_val)] = [
                        epoch + 1,
                        acc_knn_hbo_sub_1_val,
                        acc_knn_hbo_sub_2_val,
                        acc_knn_hbo_sub_4_val,
                        acc_knn_hbo_sub_6_val,
                        acc_knn_hbo_sub_8_val,
                        acc_knn_hbo_sub_10_val,
                        acc_knn_hbo_sub_12_val,
                        acc_knn_hbo_sub_14_val,
                        acc_knn_hbo_sub_16_val,
                        acc_knn_hbo_sub_18_val,
                        acc_knn_hbo_sub_20_val,
                        acc_knn_vote_val,
                        wd_val,
                        temperature,
                    ]
                    # save the results interim
                    dfs_results_knn_val.to_csv(f"{folder_models}/results_knn_subject_{subj}_val.csv", index=False)

                    # append the df values here
                    dfs_results_knn.loc[len(dfs_results_knn)] = [
                        epoch + 1,
                        acc_knn_hbo_sub_1,
                        acc_knn_hbo_sub_2,
                        acc_knn_hbo_sub_4,
                        acc_knn_hbo_sub_6,
                        acc_knn_hbo_sub_8,
                        acc_knn_hbo_sub_10,
                        acc_knn_hbo_sub_12,
                        acc_knn_hbo_sub_14,
                        acc_knn_hbo_sub_16,
                        acc_knn_hbo_sub_18,
                        acc_knn_hbo_sub_20,
                        acc_knn_vote,
                        wd_val,
                        temperature_self,
                    ]
                    # save the results interim
                    dfs_results_knn.to_csv(f"{folder_models}/results_knn_subject_{subj}.csv", index=False)

                    logger.info(
                        f"The results for hbo positive/negative val epoch {epoch + 1} knn are: Acc_k1={acc_knn_hbo_sub_1_val}, Acc_k2={acc_knn_hbo_sub_2_val}, Acc_k4={acc_knn_hbo_sub_4_val}, Acc_k8={acc_knn_hbo_sub_8_val}, Acc_k10={acc_knn_hbo_sub_10_val}, Acc_k12={acc_knn_hbo_sub_12_val},  Acc_k14={acc_knn_hbo_sub_14_val}, Acc_k16={acc_knn_hbo_sub_16_val}, Acc_k18={acc_knn_hbo_sub_18_val}, Acc_k20={acc_knn_hbo_sub_20_val}, Acc_vote={acc_knn_vote_val}."
                    )
                    logger.info(
                        f"The results for hbo positive/negative test epoch {epoch + 1} knn are: Acc_k1={acc_knn_hbo_sub_1}, Acc_k2={acc_knn_hbo_sub_2}, Acc_k4={acc_knn_hbo_sub_4}, Acc_k8={acc_knn_hbo_sub_8}, Acc_k10={acc_knn_hbo_sub_10}, Acc_k12={acc_knn_hbo_sub_12},  Acc_k14={acc_knn_hbo_sub_14}, Acc_k16={acc_knn_hbo_sub_16}, Acc_k18={acc_knn_hbo_sub_18}, Acc_k20={acc_knn_hbo_sub_20}, Acc_vote={acc_knn_vote}."
                    )

                    # apply svm 0.01 here
                    classifier_svm_hbo_1 = SVC(kernel="rbf", C=0.01, gamma="scale")
                    classifier_svm_hbo_1.fit(feat_val_sel, labels_val_use)
                    predictions_svm_hbo_1_val = classifier_svm_hbo_1.predict(feat_VAL_sel)
                    acc_svm_hbo_sub_1_val = accuracy_score(labels_VAL_use[labels_VAL_sub_use == subj], predictions_svm_hbo_1_val[labels_VAL_sub_use == subj])
                    predictions_svm_hbo_1 = classifier_svm_hbo_1.predict(feat_test_sel)
                    acc_svm_hbo_sub_1 = accuracy_score(labels_test_use[labels_test_sub_use == subj], predictions_svm_hbo_1[labels_test_sub_use == subj])

                    # apply svm 0.1
                    classifier_svm_hbo_2 = SVC(kernel="rbf", C=0.1, gamma="scale")
                    classifier_svm_hbo_2.fit(feat_val_sel, labels_val_use)
                    predictions_svm_hbo_2_val = classifier_svm_hbo_2.predict(feat_VAL_sel)
                    acc_svm_hbo_sub_2_val = accuracy_score(labels_VAL_use[labels_VAL_sub_use == subj], predictions_svm_hbo_2_val[labels_VAL_sub_use == subj])
                    predictions_svm_hbo_2 = classifier_svm_hbo_2.predict(feat_test_sel)
                    acc_svm_hbo_sub_2 = accuracy_score(labels_test_use[labels_test_sub_use == subj], predictions_svm_hbo_2[labels_test_sub_use == subj])

                    # apply svm 0.5
                    classifier_svm_hbo_3 = SVC(kernel="rbf", C=0.5, gamma="scale")
                    classifier_svm_hbo_3.fit(feat_val_sel, labels_val_use)
                    predictions_svm_hbo_3_val = classifier_svm_hbo_3.predict(feat_VAL_sel)
                    acc_svm_hbo_sub_3_val = accuracy_score(labels_VAL_use[labels_VAL_sub_use == subj], predictions_svm_hbo_3_val[labels_VAL_sub_use == subj])
                    predictions_svm_hbo_3 = classifier_svm_hbo_3.predict(feat_test_sel)
                    acc_svm_hbo_sub_3 = accuracy_score(labels_test_use[labels_test_sub_use == subj], predictions_svm_hbo_3[labels_test_sub_use == subj])

                    # apply svm 1
                    classifier_svm_hbo_4 = SVC(kernel="rbf", C=1.0, gamma="scale")
                    classifier_svm_hbo_4.fit(feat_val_sel, labels_val_use)
                    predictions_svm_hbo_4_val = classifier_svm_hbo_4.predict(feat_VAL_sel)
                    acc_svm_hbo_sub_4_val = accuracy_score(labels_VAL_use[labels_VAL_sub_use == subj], predictions_svm_hbo_4_val[labels_VAL_sub_use == subj])
                    predictions_svm_hbo_4 = classifier_svm_hbo_4.predict(feat_test_sel)
                    acc_svm_hbo_sub_4 = accuracy_score(labels_test_use[labels_test_sub_use == subj], predictions_svm_hbo_4[labels_test_sub_use == subj])

                    # apply majority voting here
                    class_voting_svm = VotingClassifier(
                        estimators=[("s1", SVC(kernel="rbf", C=0.01, gamma="scale")), ("s2", SVC(kernel="rbf", C=0.1, gamma="scale")), ("s3", SVC(kernel="rbf", C=0.5, gamma="scale")), ("s4", SVC(kernel="rbf", C=1.0, gamma="scale"))], voting="hard"
                    )
                    class_voting_svm.fit(feat_val_sel, labels_val_use)
                    predictions_svm_hbo_vote_val = class_voting_svm.predict(feat_VAL_sel)
                    acc_svm_vote_val = accuracy_score(labels_VAL_use[labels_VAL_sub_use == subj], predictions_svm_hbo_vote_val[labels_VAL_sub_use == subj])
                    predictions_svm_hbo_vote = class_voting_svm.predict(feat_test_sel)
                    acc_svm_vote = accuracy_score(labels_test_use[labels_test_sub_use == subj], predictions_svm_hbo_vote[labels_test_sub_use == subj])

                    # append the df values here
                    dfs_results_svm_val.loc[len(dfs_results_svm_val)] = [epoch + 1, acc_svm_hbo_sub_1_val, acc_svm_hbo_sub_2_val, acc_svm_hbo_sub_3_val, acc_svm_hbo_sub_4_val, acc_svm_vote_val, wd_val, temperature]
                    # save the results interim
                    dfs_results_svm_val.to_csv(f"{folder_models}/results_svm_subject_{subj}_val.csv", index=False)

                    # append the df values here
                    dfs_results_svm.loc[len(dfs_results_svm)] = [epoch + 1, acc_svm_hbo_sub_1, acc_svm_hbo_sub_2, acc_svm_hbo_sub_3, acc_svm_hbo_sub_4, acc_svm_vote, wd_val, temperature_self]
                    # save the results interim
                    dfs_results_svm.to_csv(f"{folder_models}/results_svm_subject_{subj}.csv", index=False)

                    logger.info(f"The results for hbo positive/negative epoch val {epoch + 1} svm are: Acc_s0.01={acc_svm_hbo_sub_1_val}, Acc_s0.1={acc_svm_hbo_sub_2_val}, Acc_s0.5={acc_svm_hbo_sub_3_val}, Acc_s1={acc_svm_hbo_sub_4_val}, Acc_vote={acc_svm_vote_val}")
                    logger.info(f"The results for hbo positive/negative epoch test {epoch + 1} svm are: Acc_s0.01={acc_svm_hbo_sub_1}, Acc_s0.1={acc_svm_hbo_sub_2}, Acc_s0.5={acc_svm_hbo_sub_3}, Acc_s1={acc_svm_hbo_sub_4}, Acc_vote={acc_svm_vote}")

                    # only save the models when the accuracy is higher than any of the ones registered before
                    # second the finetuned model
                    if dfs_results_knn["acc_k1"].max() <= acc_knn_hbo_sub_1:
                        # saved the torch models after training across epochs
                        # first the encoders
                        torch.save(
                            {
                                "epoch": epoch + 1,
                                "model_state_dict": encoder_EEG.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "loss": loss_general,
                            },
                            f"{folder_models}/EEG_encoder_epoch_{epoch + 1}_{subj}.pth",
                        )

                        torch.save(
                            {
                                "epoch": epoch + 1,
                                "model_state_dict": encoder_EEG_trans.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "loss": loss_general,
                            },
                            f"{folder_models}/EEG_trans_encoder_epoch_{epoch + 1}_{subj}.pth",
                        )

                        torch.save(
                            {
                                "epoch": epoch + 1,
                                "model_state_dict": classifier.state_dict(),
                                "optimizer_state_dict": optimizer_classifier.state_dict(),
                                "loss": loss_general,
                            },
                            f"{folder_models}/classifier_epoch_{epoch + 1}_{subj}.pth",
                        )

                    # here save the predictions for each model for evaluation
                    np.savez_compressed(
                        f"{folder_models}/predictions_hbo_{subj}_{epoch + 1}.npz",
                        labels=labels_test_use[labels_test_sub_use == subj],
                        pred_knn_2=predictions_knn_hbo_2,
                        pred_knn_4=predictions_knn_hbo_4,
                        pred_knn_6=predictions_knn_hbo_6,
                        pred_knn_8=predictions_knn_hbo_8,
                        pred_knn_10=predictions_knn_hbo_10,
                        pred_knn_12=predictions_knn_hbo_12,
                        pred_knn_14=predictions_knn_hbo_14,
                        pred_knn_16=predictions_knn_hbo_16,
                        pred_knn_18=predictions_knn_hbo_18,
                        pred_knn_20=predictions_knn_hbo_20,
                        predictions_knn_vote=predictions_knn_hbo_vote,
                        pred_svm_001=predictions_svm_hbo_1,
                        pred_svm_01=predictions_svm_hbo_2,
                        pred_svm_05=predictions_svm_hbo_3,
                        pred_svm_1=predictions_svm_hbo_4,
                        predictions_svm_vote=predictions_svm_hbo_vote,
                        pred_mlp_50=predictions_mlp_hbo_1,
                        pred_mlp_100=predictions_mlp_hbo_2,
                        pred_mlp_200=predictions_mlp_hbo_3,
                        pred_mlp_vote=predictions_mlp_hbo_vote,
                        labels_values=np.where(labels_test_sub_use == subj)[0],
                    )

                    if epoch + 1 >= 400:  # DO THE EVALUATION UNTIL 400 ITERATIONS**
                        # break the entire execution when reach 400 iterations - as minimum breaking iterations
                        break

            # clean the cache here on each evaluation for each self-supervised epoch..
            # save here the last interpretability map to check reliability of the measures
            np.savez_compressed(f"{folder_models}/interpretability_measures_{subj}.npz", EEG_xai=EEG_xai)

        if fold_inner >= 0:
            break


# start the main function execution here..

# define the transformations here to apply a new dataset
common_transforms = T.Compose(
    [
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        AddGaussianNoise(0.0, 0.05),  # mean=0, std=0.05
    ]
)

# get here the selector of the fNIRs type of data
fnirs_sel = int(sys.argv[1])

# read the pickle file with the EEG data from the preprocessed task related to the initial code in fNIRs_datasets/Steinmetzger_et_al
folder_path = "/work/pi_yalda_shahriari_uri_edu/JMM/fNIRS_datasets/Steinmetzger_et_al/DATA/"
save_results_path = "/scratch3/workspace/juan_mayortorres_uri_edu-temporal_results_grid_search/"

# read here the info of the EEG data
info = load_info_from_file(data_folder="/work/pi_yalda_shahriari_uri_edu/JMM/fNIRS_datasets/Steinmetzger_et_al/EEG data/2019-01-25/2019-01-25_001/EEG-64-000049.mat")

# read the fnirs info here
raw = read_raw_snirf("/work/pi_yalda_shahriari_uri_edu/JMM/fNIRS_datasets/Steinmetzger_et_al/fNIRS data/2019-01-27/2019-01-27_001/NIRS-2019-01-27_001.snirf", preload=True)

# read the fnirs data to get the subject list taken from the desktop
data_fNIRs = np.load(f"{folder_path}/fNIRs/processed_data_tddr_stim_wise.npz")

subject = list(data_fNIRs["subjects"])

info_fnirs = raw.info

# fix the info fnirs here before do the plotting take into accout that
"""
 Homer assumes the origin (0,0,0) at Cz and Y grows forward (nose direction).
 MNE assumes the origin at Cz but Y grows backward. So here we must do the correction before plotting
 In Homer, X+ points left.
 In MNE, X+ points right.

"""

montage_tmp = info_fnirs.get_montage()
chs_fnirs = montage_tmp.get_positions()["ch_pos"]

# flip y for each channel
for _, xyz in chs_fnirs.items():
    xyz[0] = -xyz[0]
    xyz[1] = -xyz[1]

# apply corrected montage
info_fnirs.set_montage(montage_tmp)

index_ground_truth_positive = [1, 3, 5, 7, 9, 11, 14, 17, 18, 20]
index_ground_truth_negative = [2, 4, 6, 8, 10, 12, 13, 14, 15, 16, 19]

indices_hbo_positive, indices_hbo_negative = reading_ground_truth_index(
    index_ground_truth_positive=index_ground_truth_positive, index_ground_truth_negative=index_ground_truth_negative, folder_data="/work/pi_yalda_shahriari_uri_edu/JMM/fNIRS_datasets//Steinmetzger_et_al/", subject=subject
)

channels_all = np.arange(0, 59, 1)
channels_all_fnirs = np.arange(0, 88, 1)

channels_removed_index = [
    channels.index("Fp1"),
    channels.index("Fp2"),
    channels.index("T7"),
    channels.index("TP9"),
    channels.index("T8"),
    channels.index("AF3"),
    channels.index("AFz"),
    channels.index("FT7"),
    channels.index("AF4"),
    channels.index("FT10"),
    channels.index("FT9"),
    channels.index("F9"),
    channels.index("FT8"),
    channels.index("F7"),
    channels.index("F8"),
    channels.index("F10"),
]

channels_picked = channels_all[~np.isin(channels_all, channels_removed_index)]
channels_picked_fnirs = channels_all_fnirs[~np.isin(channels_all_fnirs, [0, 18, 22, 33, 35, 42, 44, 62, 66, 77, 79, 86])]

# modify the info based on mne picks
reduced_info = mne.pick_info(info, channels_picked)

reduced_info_fnirs = mne.pick_info(info_fnirs, channels_picked_fnirs)

# read the EEG data first
with open(folder_path + "EEG/EEG_pre_processed.pkl", "rb") as file_EEG:
    DATA = pickle.load(file_EEG)
    EEG_no_pitch = DATA[0]
    EEG_strong_pitch_dynamic = DATA[1]
    EEG_strong_pitch_static = DATA[2]
    EEG_weak_pitch_dynamic = DATA[3]
    EEG_weak_pitch_static = DATA[4]
    subjects = DATA[5]

logger.info("EEG full data file has been read!!")

# read fNIRs file here second..
if fnirs_sel == 0:
    suffix = "tddr"
    data_fNIRS_file = np.load(folder_path + "fNIRs/processed_data_tddr_stim_wise.npz")
elif fnirs_sel == 1:
    suffix = "kalman"
    data_fNIRS_file = np.load(folder_path + "fNIRs/processed_data_kalman_stim_wise.npz")
elif fnirs_sel == 2:
    suffix = "cbs_i"
    data_fNIRS_file = np.load(folder_path + "fNIRs/processed_data_cbs_i_stim_wise.npz")

logger.info("fNIRs full data file has been read!!")
data_fNIRS = dict(data_fNIRS_file)

# get the fnirs data
fNIRS_no_pitch = data_fNIRS["data_no_pitch_stim"]
fNIRS_strong_pitch_dynamic = data_fNIRS["data_strong_pitch_dynamic_stim"]
fNIRS_strong_pitch_static = data_fNIRS["data_strong_pitch_static_stim"]
fNIRS_weak_pitch_dynamic = data_fNIRS["data_weak_pitch_dynamic_stim"]
fNIRS_weak_pitch_static = data_fNIRS["data_weak_pitch_static_stim"]
subjects_fnirs = data_fNIRS["subjects"]

fNIRS_no_pitch_def = []
# align fnirs and EEG data here
for subjects_index in range(0, len(EEG_no_pitch)):
    data_fnirs_temp = np.zeros((np.array(EEG_no_pitch[subjects_index]).shape[0], fNIRS_no_pitch.shape[1], fNIRS_no_pitch.shape[2]))
    data_fnirs_temp = fNIRS_no_pitch[subjects_index * 180 : subjects_index * 180 + np.array(EEG_no_pitch[subjects_index]).shape[0], ::]
    fNIRS_no_pitch_def.append(data_fnirs_temp)
    # transform list to numpy here
    EEG_no_pitch[subjects_index] = np.array(EEG_no_pitch[subjects_index])

fNIRS_strong_pitch_dynamic_def = []
# align fnirs and EEG data here
for subjects_index in range(0, len(EEG_strong_pitch_dynamic)):
    data_fnirs_temp = np.zeros((np.array(EEG_strong_pitch_dynamic[subjects_index]).shape[0], fNIRS_strong_pitch_dynamic.shape[1], fNIRS_strong_pitch_dynamic.shape[2]))
    data_fnirs_temp = fNIRS_strong_pitch_dynamic[subjects_index * 180 : subjects_index * 180 + np.array(EEG_strong_pitch_dynamic[subjects_index]).shape[0], ::]
    fNIRS_strong_pitch_dynamic_def.append(data_fnirs_temp)
    # transform list to numpy here
    EEG_strong_pitch_dynamic[subjects_index] = np.array(EEG_strong_pitch_dynamic[subjects_index])


fNIRS_strong_pitch_static_def = []
# align fnirs and EEG data here
for subjects_index in range(0, len(EEG_strong_pitch_static)):
    data_fnirs_temp = np.zeros((np.array(EEG_strong_pitch_static[subjects_index]).shape[0], fNIRS_strong_pitch_static.shape[1], fNIRS_strong_pitch_static.shape[2]))
    data_fnirs_temp = fNIRS_strong_pitch_static[subjects_index * 180 : subjects_index * 180 + np.array(EEG_strong_pitch_static[subjects_index]).shape[0], ::]
    fNIRS_strong_pitch_static_def.append(data_fnirs_temp)
    # transform list to numpy here
    EEG_strong_pitch_static[subjects_index] = np.array(EEG_strong_pitch_static[subjects_index])

fNIRS_weak_pitch_dynamic_def = []
# align fnirs and EEG data here
for subjects_index in range(0, len(EEG_weak_pitch_dynamic)):
    data_fnirs_temp = np.zeros((np.array(EEG_weak_pitch_dynamic[subjects_index]).shape[0], fNIRS_weak_pitch_dynamic.shape[1], fNIRS_weak_pitch_dynamic.shape[2]))
    data_fnirs_temp = fNIRS_weak_pitch_dynamic[subjects_index * 180 : subjects_index * 180 + np.array(EEG_weak_pitch_dynamic[subjects_index]).shape[0], ::]
    fNIRS_weak_pitch_dynamic_def.append(data_fnirs_temp)
    # transform list to numpy here
    EEG_weak_pitch_dynamic[subjects_index] = np.array(EEG_weak_pitch_dynamic[subjects_index])

fNIRS_weak_pitch_static_def = []
# align fnirs and EEG data here
for subjects_index in range(0, len(EEG_weak_pitch_static)):
    data_fnirs_temp = np.zeros((np.array(EEG_weak_pitch_static[subjects_index]).shape[0], fNIRS_weak_pitch_static.shape[1], fNIRS_weak_pitch_static.shape[2]))
    data_fnirs_temp = fNIRS_weak_pitch_static[subjects_index * 180 : subjects_index * 180 + np.array(EEG_weak_pitch_static[subjects_index]).shape[0], ::]
    fNIRS_weak_pitch_static_def.append(data_fnirs_temp)
    # define the CLIP dataset here for getting the EEG and fNIRs images normalized from here
    EEG_weak_pitch_static[subjects_index] = np.array(EEG_weak_pitch_static[subjects_index])

# get the embedding dims here..
embed_dim = int(sys.argv[2])
batch_size = int(sys.argv[3])
num_epochs = int(sys.argv[4])
learning_rate = float(sys.argv[5])
temperature = float(sys.argv[6])
n_folding_top = int(sys.argv[7])
interpretability_map = int(sys.argv[8])
subject_start = int(sys.argv[9])
components_tsne = int(sys.argv[10])
n_subs_val = int(sys.argv[11])
k_value_self = int(sys.argv[12])
k_value_class = int(sys.argv[13])
feat = int(sys.argv[14])

if interpretability_map == 0:
    # use the random erasing
    suffix_interp = "random_erasing"
elif interpretability_map == 1:
    suffix_interp = "ROI_based_erasing"
elif interpretability_map == 2:
    suffix_interp = "ROI_granular_based_erasing"
else:
    suffix_interp = "interaction_CAM"

os.makedirs(
    f"{save_results_path}figs_tsne_{suffix}_{embed_dim}_{batch_size}_{num_epochs}_{learning_rate}_{temperature}_stratified_cross_val_chan_remove_{suffix_interp}_{components_tsne}_{n_subs_val}_{k_value_self}_{k_value_class}_complete_class_subject_OUT_DEF_DEF_mix_more_{feat}_fix_locations_EEG_only",
    exist_ok=True,
)
os.makedirs(
    f"{save_results_path}models_{suffix}_{embed_dim}_{batch_size}_{num_epochs}_{learning_rate}_{temperature}_stratified_cross_val_chan_remove_{suffix_interp}_{components_tsne}_{n_subs_val}_{k_value_self}_{k_value_class}_complete_class_subject_OUT_DEF_DEF_mix_more_{feat}_fix_locations_EEG_only",
    exist_ok=True,
)

folder_name_save = f"{save_results_path}figs_tsne_{suffix}_{embed_dim}_{batch_size}_{num_epochs}_{learning_rate}_{temperature}_stratified_cross_val_chan_remove_{suffix_interp}_{components_tsne}_{n_subs_val}_{k_value_self}_{k_value_class}_complete_class_subject_OUT_DEF_DEF_mix_more_{feat}_fix_locations_EEG_only"
folder_models = f"{save_results_path}models_{suffix}_{embed_dim}_{batch_size}_{num_epochs}_{learning_rate}_{temperature}_stratified_cross_val_chan_remove_{suffix_interp}_{components_tsne}_{n_subs_val}_{k_value_self}_{k_value_class}_complete_class_subject_OUT_DEF_DEF_mix_more_{feat}_fix_locations_EEG_only"

"""
  execute this command using the following format: python EEG_fNIRs_clip_prelim.py <fnirs_selector> <embed_dim> <batch_size> <num_epochs_train> <learning_rate> <temperature> <folding_rate> <interpretability option> <subject_start> <components_tsne> <n_subs_val> <k_value_self>  <k_value_class> -- keep this invoking command for the subsequent evaluations
"""

logger.info("Datasets definition has been started..")

# define the stratified dataset across the whole dataset

EEG_data = np.concatenate(
    (np.concatenate(EEG_no_pitch[:], axis=0), np.concatenate(EEG_strong_pitch_dynamic[:], axis=0), np.concatenate(EEG_strong_pitch_static[:], axis=0), np.concatenate(EEG_weak_pitch_dynamic[:], axis=0), np.concatenate(EEG_weak_pitch_static[:], axis=0)), axis=0
)
fNIRs_data = np.concatenate(
    (
        np.concatenate(fNIRS_no_pitch_def[:], axis=0),
        np.concatenate(fNIRS_strong_pitch_dynamic_def[:], axis=0),
        np.concatenate(fNIRS_strong_pitch_static_def[:], axis=0),
        np.concatenate(fNIRS_weak_pitch_dynamic_def[:], axis=0),
        np.concatenate(fNIRS_weak_pitch_static_def[:], axis=0),
    ),
    axis=0,
)
labels = np.concatenate(
    (
        np.zeros((np.concatenate(fNIRS_no_pitch_def[:], axis=0).shape[0])),
        np.ones((np.concatenate(fNIRS_strong_pitch_dynamic_def[:], axis=0).shape[0])),
        2 * np.ones((np.concatenate(fNIRS_strong_pitch_static_def[:], axis=0).shape[0])),
        3 * np.ones((np.concatenate(fNIRS_weak_pitch_dynamic_def[:], axis=0).shape[0])),
        4 * np.ones((np.concatenate(fNIRS_weak_pitch_static_def[:], axis=0).shape[0])),
    ),
    axis=0,
)

EEG_data = EEG_data[0 : len(labels) - 1, :, :]
labels = labels[0 : len(labels) - 1]

labels_subjects = []

# get the subjects labels here
# create the subject label id here for training set..
for index_stimuli_category in range(0, 5):
    for index_subjects_vals in range(0, 20):
        if index_stimuli_category == 0:
            data_size_morph = EEG_no_pitch[index_subjects_vals].shape[0]
        elif index_stimuli_category == 1:
            data_size_morph = EEG_strong_pitch_dynamic[index_subjects_vals].shape[0]
        elif index_stimuli_category == 2:
            data_size_morph = EEG_strong_pitch_static[index_subjects_vals].shape[0]
        elif index_stimuli_category == 3:
            data_size_morph = EEG_weak_pitch_dynamic[index_subjects_vals].shape[0]
        elif index_stimuli_category == 4:
            data_size_morph = EEG_weak_pitch_static[index_subjects_vals].shape[0]
        labels_subjects_temp = index_subjects_vals * np.ones((data_size_morph))
        labels_subjects.append(labels_subjects_temp)

labels_subjects = np.concatenate(labels_subjects)

labels_hbo = []

# get the hbo positive and hbo negative labels here
for index_stimuli_category in range(0, 5):
    for index_subjects_vals in range(0, 20):
        if index_stimuli_category == 0:
            data_size_morph = EEG_no_pitch[index_subjects_vals].shape[0]
        elif index_stimuli_category == 1:
            data_size_morph = EEG_strong_pitch_dynamic[index_subjects_vals].shape[0]
        elif index_stimuli_category == 2:
            data_size_morph = EEG_strong_pitch_static[index_subjects_vals].shape[0]
        elif index_stimuli_category == 3:
            data_size_morph = EEG_weak_pitch_dynamic[index_subjects_vals].shape[0]
        elif index_stimuli_category == 4:
            data_size_morph = EEG_weak_pitch_static[index_subjects_vals].shape[0]

        if index_subjects_vals in indices_hbo_positive:
            labels_hbo_temp = np.ones((data_size_morph))
        else:
            labels_hbo_temp = np.zeros((data_size_morph))
        labels_hbo.append(labels_hbo_temp)

labels_hbo = np.concatenate(labels_hbo)

# defined here the stratified crossvalidation
n_splits = n_folding_top  # leave it as a variable - for keeping the good enough amount of data in the self-supervised part with enough amount of data
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


# define here the list results
results_folds_sub = np.zeros((n_splits, 8))
results_folds_hbo = np.zeros((n_splits, 8))
results_folds_stim = np.zeros((n_splits, 8))

# start the folding crossvalition here..
for fold, (train_idx, test_idx) in enumerate(skf.split(EEG_data, labels)):
    # start the experiment from a desired fold
    EEG_data_train = EEG_data[train_idx, :, :]
    EEG_data_test = EEG_data[test_idx, :, :]
    fNIRs_data_train = fNIRs_data[train_idx, :, :]
    fNIRs_data_test = fNIRs_data[test_idx, :, :]

    # USE IT FOR NOW WITH ALL THE FREQUENCIES, FILTER IT AFTER THAT TO CONTROL THE SUBJECT CLUSTERING!!
    # filtering train and test EEG data
    # logger.info("Filtering data here...")
    # for index_train_vals in range(0, EEG_data_train.shape[0]):
    #     EEG_data_train[index_train_vals, :, :] = apply_butter_worth_filter(EEG_data_train[index_train_vals, :, :], 512)

    # for index_test_vals in range(0, EEG_data_test.shape[0]):
    #     EEG_data_test[index_test_vals, :, :] = apply_butter_worth_filter(EEG_data_test[index_test_vals, :, :], 512)

    # remove the channels in the EEG side in train and test - THIS MUST BE DONE to gurantee neural meaningful results
    EEG_data_train = np.delete(EEG_data_train[:, :, :], channels_removed_index, axis=1)
    EEG_data_test = np.delete(EEG_data_test[:, :, :], channels_removed_index, axis=1)
    # remove short channels here!! This indexes were given by Steinmetzger
    fNIRs_data_train = np.delete(fNIRs_data_train[:, :, :], [0, 18, 22, 33, 35, 42, 44, 62, 66, 77, 79, 86], axis=1)
    fNIRs_data_test = np.delete(fNIRs_data_test[:, :, :], [0, 18, 22, 33, 35, 42, 44, 62, 66, 77, 79, 86], axis=1)

    labels_train = labels[train_idx]
    labels_test = labels[test_idx]
    labels_train_subj = labels_subjects[train_idx]
    labels_test_subj = labels_subjects[test_idx]
    labels_train_hbo = labels_hbo[train_idx]
    labels_test_hbo = labels_hbo[test_idx]

    # create the CLIP datasets here for the training and test
    dataset_train = CLIPDataset(
        eeg=EEG_data_train, fnirs=fNIRs_data_train, labels=labels_train, subject_labels=labels_train_subj, labels_HbO=labels_train_hbo, normalization_function=normalization_trial_minmax, eeg_transform=common_transforms, fnirs_transform=common_transforms
    )
    dataset_test = CLIPDataset(
        eeg=EEG_data_test, fnirs=fNIRs_data_test, labels=labels_test, subject_labels=labels_test_subj, labels_HbO=labels_test_hbo, normalization_function=normalization_trial_minmax, eeg_transform=common_transforms, fnirs_transform=common_transforms
    )

    logger.info("Contrastive Learning Representation training phase has started!!")

    # start here the training phase per subject
    CLIP_train(
        dataset_train=dataset_train,
        test_dataset=dataset_test,
        labels_Subject=labels_train_subj,
        num_epochs=num_epochs,
        temperature_ini=temperature,
        folder_name_save=folder_name_save,
        folder_models=folder_models,
        batch_size=batch_size,
        embed_dim=embed_dim,
        info=reduced_info,
        info_fnirs=reduced_info_fnirs,
        interp_option=interpretability_map,
        learning_rate=learning_rate,
        subject_start=subject_start,
        components_tsne=components_tsne,
        n_subs_val=n_subs_val,
        k_value_self=k_value_self,
        k_value_class=k_value_class,
        umap_feat=feat,
    )

    # Do this always for a single fold - always with the same sedd
    logger.info("Finishing task!!")
    sys.exit(0)

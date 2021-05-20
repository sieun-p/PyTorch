import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils

import numpy as np

# mne imports
import mne
from mne import io
from mne.datasets import sample

# tools for plotting confusion matrices
from matplotlib import pyplot as plt

# EEGNet-specific imports
from tensorflow.keras.callbacks import ModelCheckpoint
from EEGModel_torch import model

data_path = sample.data_path()
data = []

# Set parameters and read data
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
tmin, tmax = -0., 1
event_id = dict(aud_l=1, aud_r=2, vis_l=3, vis_r=4)
raw = io.Raw(raw_fname, preload=True, verbose=False)
raw.filter(2, None, method='iir')  # replace baselining with high-pass
events = mne.read_events(event_fname)

raw.info['bads'] = ['MEG 2443']  # set bad channels
picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')

epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=False,
                    picks=picks, baseline=None, preload=True, verbose=False)
labels = epochs.events[:, -1]

# extract raw data. scale by 1000 due to scaling sensitivity in deep learning
X = epochs.get_data()*1000 # format is in (trials, channels, samples)
y = labels

kernels, chans, samples = 1, 60, 151

# take 50/25/25 percent of the data to train/validate/test
X_train      = X[0:144,]
Y_train      = y[0:144]
X_validate   = X[144:216,]
Y_validate   = y[144:216]
X_test       = X[216:,]
Y_test       = y[216:]

# Numpy array to Tensor
X_train = torch.Tensor(X_train)
Y_train = torch.Tensor(Y_train)
Y_train = F.one_hot(Y_train.to(torch.int64)-1, 4)

X_validate = torch.Tensor(X_validate)
Y_validate = torch.Tensor(Y_validate)
Y_validate = F.one_hot(Y_validate.to(torch.int64)-1, 4)

X_test = torch.Tensor(X_test)
Y_test = torch.Tensor(Y_test)
Y_test = F.one_hot(Y_test.to(torch.int64)-1, 4)

X_train = X_train.reshape(X_train.shape[0], kernels, chans, samples)
X_validate = X_validate.reshape(X_validate.shape[0], kernels, chans, samples)
X_test = X_test.reshape(X_test.shape[0], kernels, chans, samples)
   
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

trn = data_utils.TensorDataset(X_train, Y_train)
trn_loader = data_utils.DataLoader(trn, batch_size=16, shuffle=True)

val = data_utils.TensorDataset(X_validate, Y_validate)
val_loader = data_utils.DataLoader(val, batch_size=16, shuffle=True)

test = data_utils.TensorDataset(X_test, Y_test)
test_loader = data_utils.DataLoader(test, batch_size=16, shuffle=True)

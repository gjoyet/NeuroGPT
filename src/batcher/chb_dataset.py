import os
import pdb
import numpy as np
from batcher.base import EEGDataset
from scipy.io import loadmat
from scipy.signal import butter, filtfilt


class CHBDataset(EEGDataset):
    def __init__(self, filenames, sample_keys, chunk_len=500, num_chunks=10, ovlp=50, root_path="", gpt_only=True):
        super().__init__(filenames, sample_keys, chunk_len, num_chunks, ovlp, root_path=root_path, gpt_only=gpt_only)

        trials_all = []
        labels_all = []
        total_num = []
        for fn in self.filenames:
            data = np.load(fn, mmap_mode='r')
            trials_all.append(data['epochs'])  # TODO: maybe discard the first 1000ms?
            labels_all.append(data['labels'])
            total_num.append(len(data['labels']))

        # Choices
        self.labels_string2int = {'left': 0, 'right': 1}
        self.Fs = 250  # 250Hz from original paper

        self.trials = np.vstack(trials_all)
        self.labels = np.array(labels_all).flatten()
        self.num_trials_per_sub = total_num

    def __len__(self):
        return sum(self.num_trials_per_sub)

    def __getitem__(self, idx):
        return self.preprocess_sample(self.trials[idx], self.num_chunks, self.labels[idx])

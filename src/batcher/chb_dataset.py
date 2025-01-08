import os

import h5py
import numpy as np
from batcher.base import EEGDataset


# npz Dataset
class CHBDataset_NPZ(EEGDataset):
    def __init__(self, filenames, sample_keys, chunk_len=500, num_chunks=10, ovlp=50, root_path="", gpt_only=True, num_subjects=-1):
        super().__init__(filenames, sample_keys, chunk_len, num_chunks, ovlp, root_path=root_path, gpt_only=gpt_only, num_subjects=num_subjects)

        trials_all = []
        labels_all = []
        total_num = []
        for fn in self.filenames:
            data = np.load(os.path.join(root_path, fn), mmap_mode='r')  # also works with just np.load(fn, ...)
            trials_all.append(data['epochs'][:, :, 551:2251])
            labels_all.extend(data['labels'])
            total_num.append(len(data['labels']))

        # Choices
        self.labels_string2int = {'left': 0, 'right': 1}
        self.Fs = 1000  # 250Hz from original paper

        self.trials = self.normalize(np.vstack(trials_all))
        self.labels = np.array(labels_all)
        self.num_trials_per_sub = total_num

    def __len__(self):
        return sum(self.num_trials_per_sub)

    def __getitem__(self, idx):
        return self.preprocess_sample(self.trials[idx], self.num_chunks, self.labels[idx])


# hdf5 Dataset (does not work because hdf5 objects cannot be pickled).
class CHBDataset_HDF5(EEGDataset):
    def __init__(self, filenames, sample_keys, chunk_len=500, num_chunks=10, ovlp=50, root_path="", gpt_only=True, num_subjects=-1, first_chunk_idx=501):
        super().__init__(filenames, sample_keys, chunk_len, num_chunks, ovlp, root_path=root_path, gpt_only=gpt_only, num_subjects=num_subjects)

        self.files = [h5py.File(os.path.join(root_path, fn), 'r') for fn in filenames]
        self.num_trials_per_sub = [len(f['labels']) for f in self.files]
        self.cumnum_trials = np.cumsum([0] + self.num_trials_per_sub)

        all_labels = []
        for f in self.files:
            all_labels.extend(f['labels'])
        print('\n@Guillaume\nOverall label mean: {}\nTotal number of samples: {}\n'.format(np.mean(all_labels), sum(self.num_trials_per_sub)))

        # Choices
        self.labels_string2int = {'left': 0, 'right': 1}
        self.Fs = 1000  # 250Hz from original paper

        self.range = (first_chunk_idx, first_chunk_idx + chunk_len * num_chunks)

    def __len__(self):
        return sum(self.num_trials_per_sub)

    def __getitem__(self, index):
        file_index = np.argwhere(self.cumnum_trials <= index).max()
        sample_index = index - np.where(self.cumnum_trials <= index, self.cumnum_trials, 0).max()

        # Calculate the result
        trial = self.files[file_index]['epochs'][sample_index, :, self.range[0]:self.range[1]]
        label = self.files[file_index]['labels'][sample_index, ...]

        return self.preprocess_sample(np.array(trial), self.num_chunks, np.array(label))

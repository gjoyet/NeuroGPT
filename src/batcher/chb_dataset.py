import h5py
import numpy as np
from batcher.base import EEGDataset


# npz Dataset
# class CHBDataset(EEGDataset):
#     def __init__(self, filenames, sample_keys, chunk_len=500, num_chunks=10, ovlp=50, root_path="", gpt_only=True):
#         super().__init__(filenames, sample_keys, chunk_len, num_chunks, ovlp, root_path=root_path, gpt_only=gpt_only)
#
#         trials_all = []
#         labels_all = []
#         total_num = []
#         for fn in self.filenames:
#             data = np.load(fn, mmap_mode='r')
#             trials_all.append(data['epochs'][:, :, 1251:2251])
#             labels_all.extend(data['labels'])
#             total_num.append(len(data['labels']))
#
#         # Choices
#         self.labels_string2int = {'left': 0, 'right': 1}
#         self.Fs = 1000  # 250Hz from original paper
#
#         self.trials = self.normalize(np.vstack(trials_all))
#         self.labels = np.array(labels_all)
#         self.num_trials_per_sub = total_num
#
#     def __len__(self):
#         return sum(self.num_trials_per_sub)
#
#     def __getitem__(self, idx):
#         return self.preprocess_sample(self.trials[idx], self.num_chunks, self.labels[idx])


# hdf5 Dataset
class CHBDataset(EEGDataset):
    def __init__(self, filenames, sample_keys, chunk_len=500, num_chunks=10, ovlp=50, root_path="", gpt_only=True):
        super().__init__(filenames, sample_keys, chunk_len, num_chunks, ovlp, root_path=root_path, gpt_only=gpt_only)

        self.files = [h5py.File(fn, 'r') for fn in filenames]
        self.num_trials_per_sub = [len(f['labels']) for f in self.files]
        self.cumnum_trials = np.cumsum([0] + self.num_trials_per_sub)

        # Choices
        self.labels_string2int = {'left': 0, 'right': 1}
        self.Fs = 1000  # 250Hz from original paper

    def __len__(self):
        return sum(self.num_trials_per_sub)

    def __getitem__(self, index):
        file_index = np.argwhere(self.cumnum_trials <= index).max()
        sample_index = index - np.where(self.cumnum_trials <= index, self.cumnum_trials, 0).max()

        # Calculate the result
        trial = self.files[file_index]['epochs'][sample_index, ...]
        label = self.files[file_index]['labels'][sample_index, ...]

        return self.preprocess_sample(np.array(trial), self.num_chunks, np.array(label))

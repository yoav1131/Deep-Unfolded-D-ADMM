from torch.utils.data import Dataset
import torch
import numpy as np


class SimulatedData(Dataset):
    def __init__(self, idx, snr):
        with open(f'C:/Users/yoav1/PycharmProjects/Thesis/data/LASSO data/data_{snr}_snr.npy', 'rb') as f:
            data = np.load(f, allow_pickle=True)
            label = np.load(f, allow_pickle=True)
        data = data[:1200]
        label = label[:1200]
        if idx >= 0.7 * data.shape[0]:
            self.x = torch.from_numpy(data[:idx, :, :])
            self.y = torch.from_numpy(label[:idx, :, :])
        else:
            self.x = torch.from_numpy(data[-idx:, :, :])
            self.y = torch.from_numpy(label[-idx:, :, :])
        self.samples_num = idx

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.samples_num

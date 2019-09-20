import h5py
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset


class MyDataSet(Dataset):

    def __init__(self, filename):
        f = h5py.File(filename, 'r')
        self.data = torch.from_numpy(f['data'].value)
        f.close()

    def __getitem__(self, lc, lp, lq, p, q, t):

        if t < lq * q or t >= len(self.data):
            raise IndexError("Index Overflow")

        return TrainingInstance(self.data, lc, lp, lq, p, q, t)

    def __len__(self):
        return len(self.data)


class TrainingInstance:

    def __init__(self, x, lc, lp, lq, p, q, t):
        self.s_c = x[t - lc: t].reshape(lc * 2, 20, 20)
        self.s_p = x[t - lp * p: t: p].reshape(lp * 2, 20, 20)
        self.s_q = x[t - lq * q: t: q].reshape(lq * 2, 20, 20)
        self.lc = lc
        self.lp = lp
        self.lq = lq
        self.x_t = x[t]


def split_data():
    train_rate = 0.7
    valid_rate = 0.2
    test_rate = 1 - train_rate - valid_rate





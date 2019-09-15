import h5py
import torch


def load_dataset(filename):
    f = h5py.File(filename, 'r')
    data = torch.from_numpy(f['data'].value)
    date = torch.from_numpy(f['date'].value)
    f.close()

    return data, date



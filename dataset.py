import h5py
import torch


class TrainingInstance:

    def __init__(self, x, lc, lp, lq, p, q, t):
        self.s_c = x[t - lc: t].reshape(lc * 2, 20, 20)
        self.s_p = x[t - lp * p: t: p].reshape(lp * 2, 20, 20)
        self.s_q = x[t - lq * q: t: q].reshape(lq * 2, 20, 20)
        self.x_t = x[t]


def load_dataset(filename):
    f = h5py.File(filename, 'r')
    data = torch.from_numpy(f['data'].value)
    date = torch.from_numpy(f['date'].value)
    f.close()

    return data, date



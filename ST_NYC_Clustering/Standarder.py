import numpy as np


np.random.seed(1337)  # for reproducibility


class Standard(object):

    def __init__(self):
        self.std = None
        self.mean = None

    def fit(self, X):
        self.std = np.std(X)
        self.mean = np.mean(X)
        print('std:', self.std, 'mean:', self.mean)

    def transform(self, X):
        X = 1. * (X - self.mean) / self.std
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        X = X * self.std + self.mean
        return X

    def get_std(self):
        return self.std

    def get_mean(self):
        return self.mean

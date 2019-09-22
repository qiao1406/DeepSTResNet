import numpy as np


np.random.seed(1337)  # for reproducibility


class MinMaxNormalization(object):
    """MinMax Normalization --> [-1, 1]
       x = (x - min) / (max - min).
       x = x * 2 - 1
    """

    def __init__(self):
        self.min_val = None
        self.max_val = None
        pass

    def fit(self, X):
        self.min_val = X.min()
        self.max_val = X.max()
        print("min:", self.min_val, "max:", self.max_val)

    def transform(self, X):
        X = 1.0 * (X - self.min_val) / (self.max_val - self.min_val)
        X = X * 2.0 - 1.0
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        X = (X + 1.0) / 2.0
        X = 1.0 * X * (self.max_val - self.min_val) + self.min_val
        return X

    def maxmin(self):
        return self.max_val - self.min_val

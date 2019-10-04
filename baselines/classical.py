import numpy as np
from sklearn import metrics
from sklearn.svm import SVR


def svr(X, y, kernel='poly', degree=2, C=100, epsilon=0.1):
    svm_poly_reg = SVR(kernel, degree, C, epsilon)
    svm_poly_reg.fit(X, y)
    w = svm_poly_reg.coef_
    return w


def history_average(data, width, height, channel, recent_day):
    total_err = 0.0

    for i in range(recent_day * 48, len(data)):
        pred = np.zeros(width * height)

        for j in range(1, recent_day + 1):
            pred += data[i - j * 48][channel]

        pred /= recent_day
        err = np.sqrt(metrics.mean_squared_error(data[i][channel], pred))  # RMSE loss
        total_err += err

    return total_err / (len(data) - recent_day)

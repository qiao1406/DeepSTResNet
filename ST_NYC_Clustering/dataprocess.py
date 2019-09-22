import h5py
import numpy as np
import os
from normalization import MinMaxNormalization
import pandas as pd
from Standarder import Standard
datapath = "data/NewYork"


def load_data(T,train_day,len_test,len_recent,len_day,len_week):
    taximmn = MinMaxNormalization()
    bikemmn = MinMaxNormalization()

    data = np.zeros([48 * train_day,4, 20, 20])
    path = os.path.join(datapath, "bikeV2.h5")
    f = h5py.File(path)
    xxx = f['data'][:]
    xxx = bikemmn.fit_transform(xxx)
    for i in range(48*train_day):
        data[i][:2] = xxx[i]
    f.close()

    path = os.path.join(datapath,"taxiV2together1.h5")
    f = h5py.File(path)
    xxx = f['data'][:]
    xxx = taximmn.fit_transform(xxx)
    for i in range(48*train_day):
        data[i][2:4] = xxx[i]
    efdatapath = os.path.join(datapath,"weathers.h5")
    efdata = pd.read_hdf(efdatapath)
    efdata = efdata['2016-4-1':'2016-6-30'].values


    timestamps = f['date'][:]
    f.close()


    data_train = data
    timestamps_train = timestamps

    X_recent = []
    X_day = []
    X_week = []
    Y = []
    EF = []
    timestamps_Y = []

    depends = [[8, 7, 6, 5, 4, 3, 2, 1],
               [1 * 48 * j for j in range(1, len_day + 1)],
               [7 * 48 * j for j in range(1, len_week + 1)]]
    depends[1].reverse()
    depends[2].reverse()
    xxx = depends[0][0]

    for i in range(xxx, len(data_train)):
        x_recent = [data_train[i - j] for j in depends[0]]
        y = data_train[i][3:4]
        timestamps_y = timestamps_train[i]

        X_recent.append(x_recent)
        Y.append(y)
        timestamps_Y.append(timestamps_y)
    X_recent = np.asarray(X_recent)
    Y = np.asarray(Y)
    EF = efdata[xxx:]
    # timestamps_Y = np.asarray(timestamps_Y)
    print("X_recent shape: ", X_recent.shape,  "Y shape:", Y.shape ,    "EF shape", EF.shape)
    # 分割测试集

    XR_train,  Y_train, XR_test,  Y_test  ,EF_train ,  EF_test= \
        X_recent[:-len_test * T], Y[:-len_test * T], \
        X_recent[-len_test * T:], Y[-len_test * T:]  , EF[:-len_test * T] ,  EF[-len_test * T:]
    timestamps_train, timestamps_test = timestamps_Y[:-len_test * T], timestamps_Y[-len_test * T:]
    # X_train = []
    # X_train.append(XR_train)
    # X_test = []
    # X_test.append(XR_test)
    # xxx = np.asarray(X_train)
    # print(xxx.shape)
    X_train = XR_train
    X_test = XR_test
    return X_train, Y_train, EF_train ,X_test, Y_test, EF_test, timestamps_train, timestamps_test, taximmn, bikemmn , data


def load_data_cluster(train_day, filename, channel, width, height):

    data = np.zeros([48 * train_day, height, width])
    path = os.path.join(datapath, filename)
    f = h5py.File(path)
    temp = f['data'][:]
    for i in range(48 * train_day):
        data[i] = temp[i][channel]
    f.close()
    return data


def load_data_bases(train_day, filename, channel, slice_num, width, height, reduce, normalize=True):
    """
    加载基
    :param train_day: 时间跨度
    :param filename:
    :param channel:
    :param slice_num: 分片数
    :param width:
    :param height:
    :param reduce:
    :param normalize: 决定是否进行正则化
    :return:
    """
    bike_sta = Standard()
    #
    # bike_sta = MinMaxNormalization()
    data = np.zeros([(48 * train_day - reduce) * slice_num, height, width])
    path = os.path.join(datapath, filename)
    f = h5py.File(path)
    temp = f['data'][:]

    if normalize:
        temp = bike_sta.fit_transform(temp)

    for i in range((48 * train_day - reduce) * slice_num):
        data[i] = temp[i][channel]

    f.close()
    return data, bike_sta


def load_data_CNN_bases(train_day, filename, channel, slice_num, width, height, reduce):
    """

    :param train_day:
    :param filename:
    :param channel:
    :param slice_num:
    :param width:
    :param height:
    :param reduce:
    :return:
    """
    bike_sta = Standard()
    #
    # bike_sta = MinMaxNormalization()
    data = np.zeros([(48 * train_day-reduce) * slice_num, 1, height, width])
    path = os.path.join(datapath, filename)
    print('[load_data_CNN_bases], load file:', path)

    f = h5py.File(path)
    temp = f['data'][:]
    temp = bike_sta.fit_transform(temp)

    for i in range((48 * train_day-reduce) * slice_num):
        data[i][0][:] = temp[i][channel]

    f.close()
    return data, bike_sta


def load_data_CNN(train_day, filename, channel, width, height):

    data = np.zeros([48 * train_day,1, height, width])
    path = os.path.join(datapath, filename)
    f = h5py.File(path)
    xxx = f['data'][:]
    for i in range(48 * train_day):
        data[i][0][:] = xxx[i][channel]
    f.close()
    return data

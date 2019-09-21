from torch import nn
import torch
from torch import optim
import numpy as np
import copy
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import dataprocess
from model import Lstm3
from sklearn.decomposition import NMF

K = 100
width = 50
height = 50
attnode = 1024
lr = 0.0005
epochs = 100
len_test = 48*14
len_val = 48*10
bs = 32
device = "cuda:3"


K_List = [i for i in range(150, 1000, 50)]
K_times = 2
dataH5path = "NYCBike_50x50_1slice.h5"
baseH5path = "NYCBike_50x50_8slice_train.h5"
outputpath = "output/NMF_50x50_stander.txt"
reduce = 1
train_days = 77


def main():
    loss_func = nn.MSELoss(reduction='mean')
    data_bases, bike_sta = dataprocess.load_data_bases(train_days, baseH5path, 0, 8, width, height, reduce,
                                                       normalize=False)
    data_bases = data_bases.reshape((-1, width*height))
    model_1 = NMF(n_components=K, init='random')
    W_train = model_1.fit_transform(data_bases)
    H = model_1.components_       #bases

    #求出权重序列
    data = dataprocess.load_data_cluster(91, dataH5path, 0, width, height).reshape((-1, width*height))
    W = np.dot(data, np.linalg.pinv(H))
    model_2 = Lstm3(K, K, torch.from_numpy(H).float().to(device), height, width, device).to(device)
    opt2 = optim.Adam(model_2.parameters(), lr=lr)

    X_recent = []
    Y_ = []
    depends = [8, 7, 6, 5, 4, 3, 2, 1]
    first = depends[0]
    for i in range(first, len(W)):
        x_recent = [W[i-j] for j in depends]
        y_ = data[i]

        X_recent.append(x_recent)
        Y_.append(y_)

    X_recent = torch.from_numpy(np.asarray(X_recent)).float().to(device)
    Y_ = torch.from_numpy(np.asarray(Y_)).float().to(device)
    X_train, X_test, Y_train, Y_test = X_recent[:-len_test], X_recent[-len_test:], Y_[:-len_test], Y_[-len_test:]
    X_train, Y_train, X_valid, Y_valid = X_train[:-len_val], Y_train[:-len_val], X_train[-len_val:], Y_train[-len_val:]

    train_ds = TensorDataset(X_train, Y_train)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
    valid_ds = TensorDataset(X_valid, Y_valid)
    valid_dl = DataLoader(valid_ds, batch_size=bs*2, shuffle=True)

    best_loss, best_model_wts, best_epoch = 999, copy.deepcopy(model_2.state_dict()), -1
    for epoch in range(epochs):
        model_2.train()
        for xb, yb in train_dl:
            yb_pred = model_2(xb)
            loss = (loss_func(yb_pred, yb) + 1e-6) ** 0.5
            opt2.zero_grad()
            loss.backward()
            opt2.step()

        model_2.eval()
        with torch.no_grad():
            valid_loss = (loss_func(model_2(X_valid), Y_valid) + 1e-6) ** 0.5

        print('Epoch {}/{} ,train loss:{:.4f}  true loss:{:.4f}      valid loss:{:.4f}  val trueloss:{:.4f}'.format(
            epoch + 1, epochs, (loss + 1e-6) ** 0.5, (loss + 1e-6) ** 0.5, valid_loss, valid_loss))

        if valid_loss <= best_loss:
            best_loss, best_model_wts, best_epoch = valid_loss, copy.deepcopy(model_2.state_dict()), epoch + 1

    print('Finished Training     best val loss:{:.4f}   true loss:{:.4f}      best epoch:{}'.format(best_loss,
                                                                                                    best_loss,
                                                                                                    best_epoch))
    model_2.load_state_dict(best_model_wts)

    test_ds = TensorDataset(X_test, Y_test)
    test_dl = DataLoader(test_ds, batch_size=bs)
    model_2.eval()

    data_pred = []
    for xb, yb in test_dl:
        yb_pred = model_2(xb)
        data_pred.append(yb_pred)
    data_pred = torch.cat(data_pred, 0).float().to(device)
    data_real = Y_test
    loss = loss_func(data_pred, data_real) ** 0.5

    fwrite = open(outputpath, 'a+')
    fwrite.write('K: {}      loss: {:.4f}  true loss:{:.4f}\n'.format(K, loss, loss))
    fwrite.close()
    print('K: {}      loss: {:.4f}  true loss:{:.4f}'.format(K, loss, loss))


if __name__ == '__main__':
    for item in K_List:
        for x in range(K_times):
            K = item
            main()

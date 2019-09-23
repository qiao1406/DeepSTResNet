import copy
import dataprocess
import h5py
import numpy as np
import torch
from model import EuclidDist
from model import Lstm3
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

K = 10
width = 50
height = 50
lr = 0.0005
epochs = 300
len_test = 48 * 14
len_val = 48 * 14
bs = 32
device = "cuda:2"

K_List = [121, 122]
K_times = 2
dataH5_path = "NYCBike_50x50_1slice.h5"
baseH5_path = "NYCBike_50x50_8slice_train.h5"
output_path = "output/SVD_50x50_stander.txt"
svd_path = 'output/SVDbases'
full_days = 91  # 整个数据集的时间跨度
train_days = 77  # 训练集的时间跨度
reduce = 1
run_svd = False


def perform_svd(data):
    """
    执行 SVD 分解并将得到的矩阵存储为 h5 文件
    :param data: 待分解的矩阵
    """
    u, s, v = np.linalg.svd(data)  # bases

    f = h5py.File(svd_path, 'w')
    f.create_dataset('u', u.shape, 'f')
    f.create_dataset('s', s.shape, 'f')
    f.create_dataset('v', v.shape, 'f')

    f['u'][:] = u
    f['s'][:] = s
    f['v'][:] = v
    f.close()
    print('===' * 20 + 'Enocding Finished')


def main():
    loss_func = EuclidDist()
    data_bases, bike_sta = dataprocess.load_data_bases(train_days, baseH5_path, 0, 8, width, height, reduce)
    data_bases = data_bases.reshape((-1, width * height))

    if run_svd:
        perform_svd(data_bases)

    print('Reading u, s, v from files')
    f = h5py.File(svd_path)
    u = f['u'][:]
    s = f['s'][:]
    v = f['v'][:]
    f.close()

    H = np.dot(s[:K] * np.eye(K, K), v[:K, :])
    loss1 = loss_func(torch.from_numpy(np.dot(u[:, :2500], np.dot(s * np.eye(2500, 2500), v))).float().to(device),
                      torch.from_numpy(data_bases).float().to(device)
                      )
    loss2 = loss_func(torch.from_numpy(np.dot(u[:, :K], np.dot(s[:K] * np.eye(K, K), v[:K, :]))).float().to(device),
                      torch.from_numpy(data_bases).float().to(device)
                      )
    print("loss1:{}    loss2:{}".format(loss1, loss2))

    # 求出权重序列，这里是从整个的数据集中提取的
    data = dataprocess.load_data_cluster(full_days, dataH5_path, 0, width, height).reshape((-1, width * height))
    data = (data - bike_sta.mean) / bike_sta.std
    W = np.dot(data, np.linalg.pinv(H))
    model_2 = Lstm3(K, K, torch.from_numpy(H).float().to(device), height, width, device).to(device)
    opt2 = optim.Adam(model_2.parameters(), lr=lr)

    X_recent = []
    Y_ = []
    depends = [_ for _ in range(8, 0, -1)]
    first = depends[0]
    for i in range(first, len(W)):
        x_recent = [W[i - j] for j in depends]
        y_ = data[i]

        X_recent.append(x_recent)
        Y_.append(y_)

    # seed = 20
    # random.seed(seed)
    # random.shuffle(X_recent)
    # random.seed(seed)
    # random.shuffle(Y_)

    # 分割数据集
    X_recent = torch.from_numpy(np.asarray(X_recent)).float().to(device)
    Y_ = torch.from_numpy(np.asarray(Y_)).float().to(device)
    X_train, X_test, Y_train, Y_test = X_recent[:-len_test], X_recent[-len_test:], Y_[:-len_test], Y_[-len_test:]
    X_train, Y_train, X_valid, Y_valid = X_train[:-len_val], Y_train[:-len_val], X_train[-len_val:], Y_train[-len_val:]

    train_ds = TensorDataset(X_train, Y_train)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
    valid_ds = TensorDataset(X_valid, Y_valid)
    valid_dl = DataLoader(valid_ds, batch_size=(bs * 2), shuffle=True)

    best_loss, best_model_wts, best_epoch = 999, copy.deepcopy(model_2.state_dict()), -1
    for epoch in range(epochs):
        model_2.train()
        for xb, yb in train_dl:
            yb_pred = model_2(xb)
            loss = loss_func(yb_pred, yb)
            opt2.zero_grad()
            loss.backward()
            opt2.step()

        model_2.eval()
        with torch.no_grad():
            valid_loss = loss_func(model_2(X_valid), Y_valid)

        print('Epoch {}/{} ,train loss:{:.4f}  true loss:{:.4f}      '
              'valid loss:{:.4f}  val trueloss:{:.4f}'.format(epoch + 1,
                                                              epochs,
                                                              loss + 1e-6,
                                                              (loss + 1e-6) * bike_sta.std,
                                                              valid_loss,
                                                              valid_loss * bike_sta.std))

        if valid_loss <= best_loss:
            best_loss, best_model_wts, best_epoch = valid_loss, copy.deepcopy(model_2.state_dict()), epoch + 1

    print('Finished Training     best val loss:{:.4f}   true loss:{:.4f}  '
          'best epoch:{}'.format(best_loss, best_loss * bike_sta.std, best_epoch))
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
    loss = loss_func(data_pred, data_real)
    fwrite = open(output_path, "a+")
    fwrite.write("K: {}      loss: {:.4f}  true loss:{:.4f}\n".format(K, loss, loss * bike_sta.std))
    fwrite.close()
    print("K: {}      loss: {:.4f}  true loss:{:.4f}".format(K, loss, loss * bike_sta.std))


if __name__ == '__main__':
    for item in K_List:
        for x in range(K_times):
            K = item
            main()

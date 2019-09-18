import torch.nn.functional as F
from torch import nn
from torch import optim
import torch
from torch import optim
import time
import numpy as np
import copy
from model import SoftAttention
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import h5py
import dataprocess
import math
from model import Lstm2
from sklearn.decomposition import NMF


K = 128
width = 20
height = 20
attnode = 1024
lr = 0.001
epochs = 80
len_test = 48*14
len_val = 48*10
bs = 32
device = "cuda:3"


def get_model(bases):
    model_1 = SoftAttention(width,height,K,attnode)
    model_2 = Lstm2(K,K,bases,height,width)
    return model_1, model_2, optim.Adam(model_1.parameters(), lr=lr), optim.Adam(model_2.parameters(), lr=lr)

def get_bases(Filename):
    f = h5py.File("output/"+Filename+".h5")
    return f['data'][:]
# def loss_func():
#     return nn.MSELoss()



loss_func = nn.MSELoss( size_average = True)
data = torch.from_numpy(dataprocess.load_data_cluster(91, "bikeV2.h5", 0)).float().to(device)
bases = torch.from_numpy(get_bases("clusterNYC"+str(K)+"and100")).to(device)
model_1 ,model_2, opt1, opt2 = get_model(bases)
model_1.to(device)
model_2.to(device)
# A = []
# for epoch in range(len(data)):
#     lossmin = 9999
#     for i in range(20):
#         model_1 = SoftAttention(width,height,K,attnode).to(device)
#         opt1 = optim.Adam(model_1.parameters(), lr=lr)
#         loss = torch.ones([1]).to(device)
#         loss1 = torch.zeros([1]).to(device)
#         while loss.item() != loss1.item():
#             loss1 = loss
#             x = bases
#             y = data[epoch]
#
#             y_pred, a = model_1(x,device)
#             loss = (loss_func(y_pred, y) + 1e-6)**0.5
#             opt1.zero_grad()
#             loss.backward()
#             opt1.step()
#         if(loss.item()<lossmin):
#             lossmin = loss.item()
#             amin = a
#     A.append(amin.cpu().detach().numpy())
#     print("Data: {}      Loss: {}" .format(epoch, lossmin))
#     torch.cuda.empty_cache()
# A = np.array(A)
# print(A.shape)
# f = h5py.File("output/encoder.h5","w")
# f.create_dataset("data",A.shape,"f")
# f['data'][:] = A
# f.close()
# print("==="*20+"Enocding Finished")


f = h5py.File("output/encoder.h5")
A = f['data'][:]

X_recent = []
Y_ = []
depends = [8,7,6,5,4,3,2,1]
first = depends[0]
for i in range(first, len(A)):
    x_recent = [A[i-j] for j in depends]
    y_ = data[i]

    X_recent.append(x_recent)
    Y_.append(y_.cpu().detach().numpy())

X_recent = torch.from_numpy(np.asarray(X_recent)).to(device)
Y_ = torch.from_numpy(np.asarray(Y_)).to(device)
X_train, X_test, Y_train, Y_test = X_recent[:-len_test], X_recent[-len_test:], Y_[:-len_test], Y_[-len_test:]
X_train, Y_train, X_valid, Y_valid = X_train[:-len_val], Y_train[:-len_val], X_train[-len_val:], Y_train[-len_val:]


train_ds = TensorDataset(X_train, Y_train)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
valid_ds = TensorDataset(X_valid,Y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs*2)
for epoch in range(epochs):
    model_2.train()
    for xb,yb in train_dl:
        yb_pred = model_2(xb)
        loss = (loss_func(yb_pred, yb) + 1e-6)**0.5
        opt2.zero_grad()
        loss.backward()
        opt2.step()


    model_2.eval()
    with torch.no_grad():
        valid_loss = sum((loss_func(model_2(xb), yb) + 1e-6)**0.5 for xb,yb in valid_dl)
    print('Epoch {}/{} ,train loss: {:.4f}     valid loss {:.4f}'.format(epoch+1, epochs, loss, valid_loss/len(valid_dl)))

print("Finished Training")

test_ds = TensorDataset(X_test, Y_test)
test_dl = DataLoader(test_ds, batch_size=bs)
model_2.eval()
data_pred = []
for xb, yb in test_dl:
    yb_pred = model_2(xb)
    data_pred.append(yb_pred.cpu().detach().numpy())
data_pred = torch.from_numpy(np.concatenate(data_pred,axis = 0)).to(device)
data_real = data[-len_test:]
loss = loss_func(data_pred, data_real) ** 0.5
print(loss)
from torch import nn
import torch
from torch import optim
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import dataprocess
from model import Lstm4
from sklearn.decomposition import NMF


K = 50
width = 20
height = 20
attnode = 1024
lr = 0.001
epochs = 100
len_test = 48*14
len_val = 48*10
bs = 32
device = "cuda:2"


loss_func = nn.MSELoss(reduction='mean')
data = dataprocess.load_data_cluster(91, "bikeV2.h5", 0)

data2 = data.reshape((91*48,-1))

model_1 = NMF(n_components = K, init='random')
W = model_1.fit_transform(data2)
H = torch.from_numpy(model_1.components_).float().to(device)

model_2 = Lstm4(K,K,H,height,width,device)
model_2.to(device)
opt2 = optim.Adam(model_2.parameters(), lr=lr)


X_recent = []
Y_ = []
depends = [8,7,6,5,4,3,2,1]
first = depends[0]
for i in range(first, len(W)):
    x_recent = [W[i-j] for j in depends]
    y_ = W[i]

    X_recent.append(x_recent)
    Y_.append(y_)

X_recent = torch.from_numpy(np.asarray(X_recent)).float().to(device)
Y_ = torch.from_numpy(np.asarray(Y_)).float().to(device)
X_train, X_test, Y_train, Y_test = X_recent[:-len_test], X_recent[-len_test:], Y_[:-len_test], Y_[-len_test:]
X_train, Y_train, X_valid, Y_valid = X_train[:-len_val], Y_train[:-len_val], X_train[-len_val:], Y_train[-len_val:]


train_ds = TensorDataset(X_train, Y_train)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
valid_ds = TensorDataset(X_valid,Y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs*2,shuffle=True)


for epoch in range(epochs):
    model_2.train()
    for xb,yb in train_dl:
        yb_pred = model_2(xb)
        loss = (loss_func(yb_pred, yb) + 1e-6)**0.5
        opt2.zero_grad()
        loss.backward()
        opt2.step()

    with torch.no_grad():
        valid_loss = sum((loss_func(model_2(xb), yb) + 1e-6)**0.5 for xb,yb in valid_dl)
    print('Epoch {}/{} ,train loss: {:.4f}     valid loss {:.4f}'.format(epoch+1, epochs, loss, valid_loss/len(valid_dl)))

print("Finished Training")

test_ds = TensorDataset(X_test, Y_test)
test_dl = DataLoader(test_ds, batch_size=bs)
data_pred = []
loss = sum((loss_func(model_2(xb), yb)+1e-6)**0.5 for xb, yb in test_dl)
for xb, yb in test_dl:
    yb_pred = model_2(xb)
    yb_pred = yb_pred.mm(H).view(-1,height,width)
    data_pred.append(yb_pred.cpu().detach().numpy())
data_pred = torch.from_numpy(np.concatenate(data_pred, axis=0)).float().to(device)
data_real = torch.from_numpy(data[-len_test:]).float().to(device)
loss = loss_func(data_pred, data_real) ** 0.5
print(loss)

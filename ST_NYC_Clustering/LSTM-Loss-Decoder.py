from torch import nn
import torch
from torch import optim
import numpy as np
from model import SoftAttention
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import h5py
import dataprocess
from model import Lstm



K = 128
width = 20
height = 20
attnode = 1024
lr = 0.001
epochs = 10
len_test = 48*14
len_val = 48*10
bs = 32
device = "cuda:2"


def get_model():
    model_1 = SoftAttention(width,height,K,attnode)
    model_2 = Lstm(K,K)
    return model_1, model_2, optim.Adam(model_1.parameters(), lr=lr), optim.Adam(model_2.parameters(), lr=lr)


def get_bases(Filename):
    f = h5py.File("output/"+Filename+".h5")
    return f['data'][:]
# def loss_func():
#     return nn.MSELoss()


model_1 ,model_2, opt1, opt2 = get_model()
model_1.to(device)
model_2.to(device)
loss_func = nn.MSELoss( size_average = True)
data = torch.from_numpy(dataprocess.load_data_cluster(91, "bikeV2.h5", 0)).float().to(device)
bases = torch.from_numpy(get_bases("clusterNYC"+str(K)+"and100")).to(device)
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
    y_ = A[i]

    X_recent.append(x_recent)
    Y_.append(y_)
X_recent = torch.from_numpy(np.asarray(X_recent)).to(device)
Y_ = torch.from_numpy(np.array(Y_)).to(device)
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
# loss = sum((loss_func(model_2(xb), yb)+1e-6)**0.5 for xb,yb in test_dl )
# print(loss/len(test_dl))
data_pred = []
bases = bases.view(K,-1)
for xb, yb in test_dl:
    yb_pred = model_2(xb).mm(bases).view(-1,height,width)
    data_pred.append(yb_pred.cpu().detach().numpy())
data_pred = torch.from_numpy(np.concatenate(data_pred,axis = 0)).to(device)
data_real = data[-len_test:]
loss = loss_func(data_pred, data_real) ** 0.5
print(loss)
# def train_model(model: nn.Module, dataloaders, criterion, optimizer, num_epochs=25, device=torch.device('cpu')):
#     since = time.time()
#
#     val_acc_history = []
#
#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_acc = 0.0
#
#     for epoch in range(num_epochs):
#         print('Epoch {}/{}'.format(epoch, num_epochs - 1))
#         print('-' * 10)
#
#         # Each epoch has a training and validation phase
#         for phase in ['train', 'val']:
#             if phase == 'train':
#                 model.train()  # Set model to training mode
#             else:
#                 model.eval()  # Set model to evaluate mode
#
#             running_loss = 0.0
#             running_corrects = 0
#
#             # Iterate over data.
#             for inputs, labels in dataloaders[phase]:
#                 inputs, labels = inputs.to(device), labels.to(device)
#
#                 # forward
#                 # track history if only in train
#                 with torch.set_grad_enabled(phase == 'train'):
#                     # Get model outputs and calculate loss
#                     outputs = model(inputs)
#                     loss = criterion(outputs, labels)
#
#                     _, preds = torch.max(outputs, 1)
#
#                     # backward + optimize only if in training phase
#                     if phase == 'train':
#                         optimizer.zero_grad()  # zero the parameter gradients
#                         loss.backward()
#                         optimizer.step()
#
#                 # statistics
#                 running_loss += loss.item() * inputs.size(0)
#                 running_corrects += torch.sum(preds == labels.data)
#
#             epoch_loss = running_loss / len(dataloaders[phase].dataset)
#             epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
#
#             print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
#
#             # deep copy the model
#             if phase == 'val' and epoch_acc > best_acc:
#                 best_acc = epoch_acc
#                 best_model_wts = copy.deepcopy(model.state_dict())
#             if phase == 'val':
#                 val_acc_history.append(epoch_acc)
#
#     time_elapsed = time.time() - since
#     print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
#     print('Best val Acc: {:4f}'.format(best_acc))
#
#     # load best model weights
#     model.load_state_dict(best_model_wts)
#     return model, val_acc_history
#
#
# def test_model(model, dataloader, device=torch.device('cpu')):
#     since = time.time()
#     outputs_list, labels_list = list(), list()
#
#     model.eval()  # Set model to evaluate mode
#
#     # Iterate over data.
#     for inputs, labels in dataloader:
#         inputs, labels = inputs.to(device), labels.to(device)
#
#         labels_list.append(labels.numpy())
#
#         # forward
#         # track history if only in train
#         with torch.set_grad_enabled(False):
#             outputs = model(inputs)
#             outputs_list.append(outputs.numpy())
#
#     time_elapsed = time.time() - since
#     print('Test complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
#
#     return np.concatenate(labels_list), np.concatenate(outputs_list)


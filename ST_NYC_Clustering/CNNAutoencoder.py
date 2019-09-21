import copy
import dataprocess
import h5py
import math
import numpy as np
import random
import time
import torch
import torch.nn.functional as F
from model import CNNAutoencoder, SoftAttention, Lstm6
from sklearn.decomposition import NMF
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset


# parameters
K = 121
width = 50
height = 50
lstm_height = 11
lstm_width = 11

lr = 0.0005
epochs = 100
epochs_ae = 50
len_test = 48 * 14
len_val = 48 * 14
bs = 32
device = "cuda:2"

# K_List = [i for i in range(80, 1000, 20)]
# K_times = 2
dataH5_path = 'NYCBike_50x50_1slice.h5'
baseH5_path = 'NYCBike_50x50_8slice_train.h5'
output_path = 'output/CNNAutoencoder_50x50_stander.txt'
reduce = 1
train_days = 77


def main():
    loss_func = nn.MSELoss(reduction='mean')
    data_bases, bike_sta = dataprocess.load_data_CNN_bases(train_days, baseH5_path, 0, 8, width, height, reduce)
    data_bases = torch.from_numpy(data_bases).float().to(device)

    model_1 = CNNAutoencoder().to(device)
    print(model_1)
    opt1 = optim.Adam(model_1.parameters(), lr=lr)

    ae_ds = TensorDataset(data_bases, data_bases)
    ae_dl = DataLoader(ae_ds, batch_size=(bs * 2), shuffle=True)
    for epoch in range(epochs_ae):
        for xb, yb in ae_dl:
            yb_pred = model_1(xb)
            loss = loss_func(yb, yb_pred) ** 0.5
            opt1.zero_grad()
            loss.backward()
            opt1.step()
        print("loss:{:.4f}   true loss:{:.4f}    epochs:{}".format(loss, loss * bike_sta.std, epoch))

    torch.save(model_1.encoder, "output/bases/CNN_bikepp_en")
    torch.save(model_1.decoder, "output/bases/CNN_bikepp_de")

    encoder = torch.load("output/bases/CNN_bikepp_en")
    decoder = torch.load("output/bases/CNN_bikepp_de")

    print(encoder)
    print(decoder)

    # 求出权重序列
    data = dataprocess.load_data_CNN(91, dataH5_path, 0, width, height)
    data = (data - bike_sta.mean) / bike_sta.std

    W = encoder(torch.from_numpy(data).float().to(device)).cpu().detach().numpy()
    model_2 = Lstm6(K, K, decoder, lstm_height, lstm_width, device).to(device)
    opt2 = optim.Adam(model_2.parameters(), lr=lr)

    X_recent = []
    Y_ = []
    depends = [8, 7, 6, 5, 4, 3, 2, 1]
    first = depends[0]
    for i in range(first, len(W)):
        x_recent = [W[i - j] for j in depends]
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
    valid_dl = DataLoader(valid_ds, batch_size=bs * 2, shuffle=True)

    best_loss, best_model_wts, best_epoch = 999, copy.deepcopy(model_2.state_dict()), -1
    for epoch in range(epochs):
        model_2.train()
        for xb, yb in train_dl:

            yb_pred = model_2(xb, decoder)
            loss = loss_func(yb_pred, yb)
            opt2.zero_grad()
            loss.backward()
            opt2.step()

        model_2.eval()
        with torch.no_grad():
            valid_loss = (loss_func(model_2(X_valid, decoder), Y_valid) + 1e-6) ** 0.5
        print('Epoch {}/{} ,train loss:{:.4f}  true loss:{:.4f}      valid loss:{:.4f}  val trueloss:{:.4f}'.format(
            epoch + 1, epochs, (loss + 1e-6) ** 0.5, (loss + 1e-6) ** 0.5 * bike_sta.std, valid_loss,
            valid_loss * bike_sta.std))
        if valid_loss <= best_loss:
            best_loss, best_model_wts, best_epoch = valid_loss, copy.deepcopy(model_2.state_dict()), epoch + 1

    print("Finished Training     best val loss:{:.4f}   true loss:{:.4f}      best epoch:{}".format(best_loss,
                                                                                                    best_loss * bike_sta.std,
                                                                                                    best_epoch))
    model_2.load_state_dict(best_model_wts)

    test_ds = TensorDataset(X_test, Y_test)
    test_dl = DataLoader(test_ds, batch_size=bs)
    model_2.eval()
    data_pred = []

    for xb, yb in test_dl:
        yb_pred = model_2(xb, decoder)
        data_pred.append(yb_pred)
    data_pred = torch.cat(data_pred, 0).float().to(device)
    data_real = Y_test
    loss = loss_func(data_pred, data_real) ** 0.5
    fwrite = open(output_path, "a+")
    fwrite.write("K: {}      loss: {:.4f}  true loss:{:.4f}\n".format(K, loss, loss * bike_sta.std))
    fwrite.close()
    print("K: {}      loss: {:.4f}  true loss:{:.4f}".format(K, loss, loss * bike_sta.std))


if __name__ == '__main__':
    main()

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
# def model_train(model, opt, loss_func, bikesta, train_dl, val_dl, epochs):
#     best_loss, best_model_wts, best_epoch = 999, copy.deepcopy(model.state_dict()), -1
#     for epoch in range(epochs):
#         model.train()
#         for xb, yb in train_dl:
#             yb_pred = model(xb)
#             loss = loss_func(yb_pred, yb)
#             opt.zero_grad()
#             loss.backward()
#             opt.step()
#
#         model.eval()
#         valid_loss = 0
#         with torch.no_grad():
#             for xb, yb in val_dl:
#                 valid_loss += (loss_func(model(xb), yb) + 1e-6) ** 0.5 * xb.size(0)
#
#         valid_loss = valid_loss/len(val_dl.dataset)
#         print('Epoch {}/{} ,train loss:{:.4f}  true loss:{:.4f}      valid loss:{:.4f}  val trueloss:{:.4f}'.format(
#             epoch + 1, epochs, (loss + 1e-6) ** 0.5, (loss + 1e-6) ** 0.5 * bikesta.std, valid_loss,
#             valid_loss * bikesta.std))
#         if valid_loss <= best_loss:
#             best_loss, best_model_wts, best_epoch = valid_loss, copy.deepcopy(model.state_dict()), epoch + 1
#     print("Finished Training     best val loss:{:.4f}   true loss:{:.4f}      best epoch:{}".format(best_loss,
#                                                                                                     best_loss * bikesta.std,
#                                                                                                     best_epoch))
#     model.load_state_dict(best_model_wts)
#     return model

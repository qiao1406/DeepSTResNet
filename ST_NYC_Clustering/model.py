import tensorly as tl
import torch
import torch.nn.functional as F
from torch import nn


class SoftAttention(nn.Module):

    def __init__(self, width, height, K, attnode):
        super(SoftAttention, self).__init__()
        self.lin1 = nn.Linear(K * width * height, attnode)
        self.lin2 = nn.Linear(attnode, K, bias=False)
        self.K = K

    def forward(self, x, device):
        y = x.view(-1)
        y = F.relu(self.lin1(y))
        y = F.relu(self.lin2(y))
        y = F.softmax(y)
        z = torch.zeros([20, 20]).to(device)
        for i in range(self.K):
            z += y[i] * x[i]
        return z, y


class CNNAutoencoder(nn.Module):

    def __init__(self):
        super(CNNAutoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Conv2d(1, 20, (3, 3), stride=1, padding=0),
                                     nn.Conv2d(20, 20, (2, 2), stride=2, padding=0),
                                     nn.Conv2d(20, 1, (3, 3), stride=1, padding=0),
                                     nn.Conv2d(1, 1, (2, 2), stride=2, padding=0))
        self.decoder = nn.Sequential(nn.ConvTranspose2d(1, 1, (2, 2), stride=2, padding=0),
                                     nn.ConvTranspose2d(1, 20, (3, 3), stride=1, padding=0),
                                     nn.ConvTranspose2d(20, 20, (2, 2), stride=2, padding=0),
                                     nn.ConvTranspose2d(20, 1, (3, 3), stride=1, padding=0),
                                     )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class DenseAutoencoder(nn.Module):
    def __init__(self):
        super(DenseAutoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(50 * 50, 121)


                                     )
        self.decoder = nn.Sequential(

                                     nn.Linear(121, 2500)
                                     )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class ResNet(nn.Module):

    def __init__(self, len):
        super(ResNet, self).__init__()
        self.cnn1 = nn.Conv2d()

    def forward(self, x):
        pass


class Lstm(nn.Module):
    def __init__(self, input_size, output_size):
        super(Lstm, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=output_size,
                            num_layers=1,
                            batch_first=True)

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        output_last_timestep = h_n[-1, :, :]
        return output_last_timestep


class Lstm2(nn.Module):
    def __init__(self, input_size, output_size, bases, height, width):
        super(Lstm2, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=output_size,
                            num_layers=1,
                            batch_first=True)
        self.bases = bases
        self.height = height
        self.width = width

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        output_last_timestep = h_n[-1, :, :]
        out = output_last_timestep.mm(self.bases.view(len(self.bases), -1)).view(-1, self.height, self.width)
        return out


# NMF 先deocde后loss
class Lstm3(nn.Module):
    def __init__(self, input_size, output_size, bases, height, width, device):
        super(Lstm3, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=output_size,
                            num_layers=1,
                            batch_first=True)
        self.bases = bases
        self.height = height
        self.width = width
        self.device = device

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        output_last_timestep = h_n[-1, :, :]
        out = output_last_timestep.mm(self.bases)
        return out


# NMF 先loss后decode
class Lstm4(nn.Module):
    def __init__(self, input_size, output_size, bases, height, width, device):
        super(Lstm4, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=output_size,
                            num_layers=1,
                            batch_first=True)
        self.bases = bases
        self.height = height
        self.width = width
        self.device = device

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        output_last_timestep = h_n[-1, :, :]
        out = output_last_timestep
        return out


# Tucker 先deocde后loss
class Lstm5(nn.Module):
    def __init__(self, input_size, output_size, bases, height, width, device):
        super(Lstm5, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=output_size,
                            num_layers=1,
                            batch_first=True)
        self.core = bases[0]
        self.factors = bases[1]
        self.height = height
        self.width = width
        self.device = device

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        output_last_timestep = h_n[-1, :, :]
        out = tl.tucker_to_tensor((self.core.to(self.device), [output_last_timestep, self.factors[1].to(self.device),
                                                               self.factors[2].to(self.device)]))
        return out


# for  CNN ae
class Lstm6(nn.Module):

    def __init__(self, input_size, output_size, decoder, height, width, device):
        super(Lstm6, self).__init__()
        self.decoder = decoder
        # for p in self.parameters():
        #     p.requires_grad = False

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=output_size, num_layers=1, batch_first=True)
        self.height = height
        self.width = width
        self.device = device

    def forward(self, x, decoder):
        x = x.view(x.shape[0], x.shape[1], -1)
        output, (h_n, c_n) = self.lstm(x)
        output_last_timestep = h_n[-1, :, :]
        de_input = output_last_timestep.view(-1, 1, self.height, self.width)
        output = self.decoder(de_input)
        return output


# for  Dense ae
class Lstm7(nn.Module):
    def __init__(self, input_size, output_size, encoder, decoder, height, width, device):
        super(Lstm7, self).__init__()
        self.decoder = decoder
        self.encoder = encoder
        for p in self.parameters():
            p.requires_grad = False

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=output_size, num_layers=1, batch_first=True)
        self.height = height
        self.width = width
        self.device = device

    def forward(self, x):
        # print(x.shape)
        output = self.encoder(x)
        output, (h_n, c_n) = self.lstm(output)
        output_last_timestep = h_n[-1, :, :]
        output = output_last_timestep.view(-1, self.height * self.width)
        output = self.decoder(output)
        return output


class RMSELoss(nn.Module):

    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_hat, y):
        return torch.sqrt(self.mse(y_hat, y))


class EuclidDist(nn.Module):

    def __init__(self):
        super(EuclidDist, self).__init__()

    def forward(self, y_hat, y):
        return torch.dist(y_hat, y)

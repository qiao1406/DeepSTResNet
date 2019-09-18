import torch.optim as optim
from dataset import *
from model import *


# def fusion(wc, wp, wq, xc, xp, xq):
#     return wc.mul(xc) + wp.mul(xp) + wq.mul(xq)


# def train_model(data, lc, lp, lq, p, q, t, res_layer):
#
#     train_data = TrainingInstance(data, lc, lp, lq, p, q, t)
#
#     # Init weights
#     wc = torch.ones(train_data.s_c.shape)
#     wp = torch.ones(train_data.s_p.shape)
#     wq = torch.ones(train_data.s_q.shape)
#
#     # outputs
#     xc = SharedStructure(res_layer, lc)
#     xp = SharedStructure(res_layer, lp)
#     xq = SharedStructure(res_layer, lq)
#
#     x_res = fusion(wc, wp, wq, xc, xp, xq)
#     x_pred = func.tanh(x_res)
#
#     criterion = RMSELoss()
#     loss = criterion(x_pred, train_data.x_t)
#     loss.backword()


cirterion = RMSELoss()
optimizer = optim.SGD()
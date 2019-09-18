
import random
import h5py
#
# K_List = [i for i in range(80,1000,20)]
# print(K_List)
# x = [x for x in range(20)]
# y = [y for y in range(20)]
# print(x)
# random.seed(20)
# random.shuffle(x)
# print(x)
# random.seed(20)
# random.shuffle(y)
# print(y)
f = h5py.File("output/SVDbases")
s = f['s'][:]
torch.from_numpy(data).float().to(device)
f.close()

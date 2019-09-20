import numpy as np
import dataprocess
import random
import datetime
import h5py

K = 128
Filename = "clusterNYC"+str(K)+"and100"
fazhi = 0.45


def newh5(name):
    fwrite = h5py.File("output/" + name + ".h5", "w")
    fwrite.create_dataset("data", (K, 20, 20), 'f')
    fwrite.close()


def distance(A, data):
    dis = 0
    for i in range(len(data)):
        d = 99999999999999999999999999999999999
        for j in A:
            dnew = np.linalg.norm(data[i] - data[j])
            if dnew < d:
                d = dnew
        dis += d * d
    return dis
def process(A):
    B = []
    datebegin = datetime.datetime(2016, 4, 1, 0, 0, 0)
    halfhour = datetime.timedelta(minutes=30)
    for i in A:
        B.append((datebegin + halfhour * i).strftime("%w-%H:%M"))
    return B

def findcluster(A,data,t):
    num = -1
    dis = 999999999999999999999999
    for i in range(len(A)):
        d = np.linalg.norm(data[t] - A[i])
        if d<dis:
            dis = d
            num = i
    return num

def bases():
    f = h5py.File("output/"+Filename+".h5")
    return f['data']




# 0是out     1是in    欧氏距离    k-means++   4368

A = []        #类中心
AA = []
for i in range(K):
    AA.append([])
data = dataprocess.load_data_cluster(91, "bikeV2.h5", 0)
A.append(random.randint(0, 91 * 48 - 1))

while True:
    dis = distance(A, data)
    for i in range(len(data)):
        d = 99999999999999999999999999999999999
        for j in A:
            dnew = np.linalg.norm(data[i] - data[j])
            if dnew < d:
                d = dnew
        d = d * d
        if random.random() < d / dis:
            A.append(i)
            break
        else:
            pass
    if len(A) == K:
        break

A = [data[A[i]] for i in range(len(A))]
B = []
dis = 999999999999999999999999999999999
while dis > fazhi:
    #分类
    dis = 0
    for i in range(len(data)):
        num = findcluster(A,data,i)
        AA[num].append(i)
    #计算中心
    for i in range(len(AA)):
        sum = np.zeros([20,20])
        for j in range(len(AA[i])):
            sum += data[AA[i][j]]
        newcenter = sum/len(AA[i])
        d = np.linalg.norm(A[i] - newcenter)
        dis += d
        A[i] = newcenter
    print(dis)


newh5(Filename)
count = np.array(A)
print(count)
f = h5py.File("output/" + Filename + ".h5")
f['data'][:] = count
f.close()

import h5py
import csv
import numpy as np
import pandas as pd
import datetime
# f = h5py.File("I:/NYC/CitiBike/output/NYCBike_50x50_1slice.h5")
f = h5py.File("E:\\PyCharm 2018.1.3\\workspace\\ST-NYC-Clustering\\data\\NewYork\\bikeV2.h5")

dict = {}
count = 0
x=np.sum(f['data'])
print(x)
for i in f.keys():
    if (i == "data"):
      print (i)
      print(f[i].shape)
      for j in f[i]:

        for k in j:
            for l in k:

                for key in l:
                    dict[key] = dict.get(key, 0) + 1


print(dict)

# with open('C:\\Users\\Xinran\\Desktop\\bike1020.csv', 'w') as f:  # Just use 'w' mode in 3.x
#     w = csv.DictWriter(f, dict.keys())
#     w.writeheader()
#     w.writerow(dict)

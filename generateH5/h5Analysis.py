# import pandas as pd
#
# data=pd.read_hdf("C:\\Users\\Xinran\\Desktop\\taxiV2.h5" ,key=None, mode='r')
# print(data)
import math
import h5py
import datetime
import numpy as np

maxlng = -73.92
minlng = -74.02
maxlat = 40.8
minlat = 40.67
grid = [50, 50]
gridlng = (maxlng - minlng) / grid[0]
gridlat = (maxlat - minlat) / grid[1]
halfhour = datetime.timedelta(minutes=30)
datemin = datetime.datetime(2016, 4, 1, 0, 0, 0)
datemax = datetime.datetime(2016, 6, 30, 0, 0, 0)  - halfhour
total_len = (datemax - datemin)/halfhour
print(total_len)
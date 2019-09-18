import math
import h5py
import datetime
import numpy as np


maxlng = -73.92
minlng = -74.02
maxlat = 40.8
minlat = 40.67
grid = [50,50]
gridlng = (maxlng - minlng)/grid[0]
gridlat = (maxlat - minlat)/grid[1]
datemin = datetime.datetime(2016,4,1,0,0,0)
datemax = datetime.datetime(2016,6,30,0,0,0)
halfhour = datetime.timedelta(minutes=30)


def getxy(lng,lat):
    x = math.floor((lng - minlng)/gridlng)
    y = math.floor((maxlat - lat)/gridlat)
    if (x == grid[1]):
        x = grid[1] - 1
    if (y == grid[0]):
        y = grid[0] - 1
    return x,y

def main():
    print("use old h5")
    fwrite = h5py.File("I:\\NYC\\Citi Bike\\output\\taxiV2.h5")
    count = np.zeros([4368, 2, grid[0], grid[1]])


    for month in ["2016-04","2016-05","2016-06"]:
        pathr = "I:\\NYC\\CitiBike\\yellow_tripdata_"+ month+".csv"
        print(pathr)
        f = open(pathr)
        line = f.readline()
        while(True):
            line = f.readline()
            if(not line):
                print("此文件读完")
                break
            xxx = line.split(",")
            lat = float(xxx[6])
            lng = float(xxx[5])


            date = datetime.datetime.strptime(xxx[1],"%Y-%m-%d %H:%M:%S")
            t = math.floor((date-datemin)/halfhour)
            if(lng>=minlng and lng<=maxlng and lat>=minlat and lat<=maxlat and t>=0 and t<4368):
                x,y = getxy(lng,lat)

                count[t][0][y][x] = count[t][0][y][x]+1

            lng = float(xxx[9])
            lat = float(xxx[10])
            date = datetime.datetime.strptime(xxx[2], "%Y-%m-%d %H:%M:%S")
            t = math.floor((date - datemin) / halfhour)
            if (lng>=minlng and lng<=maxlng and lat>=minlat and lat<=maxlat and t>=0 and t<4368):
                x, y = getxy(lng, lat)

                count[t][1][y][x] = count[t][1][y][x] + 1

        f.close()

   
    fwrite['data'][:] = count
    fwrite.close()

if __name__ == '__main__':
    main()
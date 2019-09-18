import math
import h5py
import datetime
import numpy as np

maxlng = -73.92
minlng = -74.02
maxlat = 40.8
minlat = 40.67
grid = [40, 40]
gridlng = (maxlng - minlng) / grid[0]
gridlat = (maxlat - minlat) / grid[1]



def getxy(lng,lat):
    x = math.floor((lng - minlng)/gridlng)
    y = math.floor((maxlat - lat)/gridlat)
    if (x == grid[1]):
        x = grid[1] - 1
    if (y == grid[0]):
        y = grid[0] - 1
    return x,y


def main():

    total_count = []

    halfhour = datetime.timedelta(minutes=30)
    datemin = datetime.datetime(2016, 4, 1, 0, 0, 0) - datetime.timedelta(minutes=4)
    datemax = datetime.datetime(2016, 7, 1, 0, 0, 0) - halfhour - datetime.timedelta(minutes=4)
    total_len = (datemax - datemin) / halfhour

    print(total_len)
    for shift in range(8):
        datamax = datemax + datetime.timedelta(minutes=4)
        datemin = datemin + datetime.timedelta(minutes=4)
        print(datemin)

        count = np.zeros([int(total_len), 2, grid[0], grid[1]])
        for month in ["201604","201605","201606"]:
            pathr = "I:\\NYC\\CitiBike\\"+month+"-citibike-tripdata.csv"
            print(pathr)
            f = open(pathr)
            line = f.readline()
            while (True):
                line = f.readline()
                if (not line):
                    print(1)
                    break
                xxx = line.split('","')
                lng = float(xxx[6])
                lat = float(xxx[5])
                date = datetime.datetime.strptime(xxx[1], "%m/%d/%Y %H:%M:%S")
                t = math.floor((date - datemin) / halfhour)
                if (lng >= minlng and lng <= maxlng and lat >= minlat and lat <= maxlat and t >= 0 and t < total_len):
                    x, y = getxy(lng, lat)
                    count[t][0][y][x] = count[t][0][y][x] + 1


                lng = float(xxx[10])
                lat = float(xxx[9])
                date = datetime.datetime.strptime(xxx[2], "%m/%d/%Y %H:%M:%S")
                t = math.floor((date - datemin) / halfhour)
                if (lng >= minlng and lng <= maxlng and lat >= minlat and lat <= maxlat and t >= 0 and t < total_len):
                    x, y = getxy(lng, lat)
                    count[t][1][y][x] = count[t][1][y][x] + 1

            f.close()

        print(count.shape)
        total_count.append(count)

    total_count = np.concatenate(total_count,axis = 0)
    print(total_count.shape)
    fwrite = h5py.File("I:/NYC/CitiBike/output/NYCBike_40x40_8slice.h5")
    fwrite['data'][:] = total_count
    fwrite.close()


if __name__ == '__main__':
    main()
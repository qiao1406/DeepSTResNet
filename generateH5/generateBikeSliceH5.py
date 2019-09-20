import math
import h5py
import datetime
import numpy as np


max_lng = -73.92
min_lng = -74.02
max_lat = 40.8
min_lat = 40.67
grid = [50, 50]
grid_lng = (max_lng - min_lng) / grid[0]
grid_lat = (max_lat - min_lat) / grid[1]


def getxy(lng, lat):
    """
    通过经纬度计算该点在矩阵中的位置
    :param lng:
    :param lat:
    :return:
    """
    x = math.floor((lng - min_lng) / grid_lng)
    y = math.floor((max_lat - lat) / grid_lat)
    if x == grid[1]:
        x = grid[1] - 1
    if y == grid[0]:
        y = grid[0] - 1
    return x, y


def main():

    total_count = []

    half_hour = datetime.timedelta(minutes=30)
    date_min = datetime.datetime(2016, 4, 1, 0, 0, 0) - datetime.timedelta(minutes=4)
    date_max = datetime.datetime(2016, 6, 17, 0, 0, 0) - half_hour - datetime.timedelta(minutes=4)
    total_len = (date_max - date_min) / half_hour

    print(total_len)
    for shift in range(8):
        date_min = date_min + datetime.timedelta(minutes=4)
        print('slice from', date_min)

        count = np.zeros([int(total_len), 2, grid[0], grid[1]])
        for month in ['201604', '201605', '201606']:
            path = 'C:\\Study and Work\\实验室\\PAPERS\\论文阅读第一期\\CitiBike\\' + month + '-citibike-tripdata.csv'
            print(path)
            f = open(path)
            line = f.readline()

            while True:
                line = f.readline()
                if not line:
                    print(1)
                    break
                line_items = line.split('","')
                lng = float(line_items[6])
                lat = float(line_items[5])
                date = datetime.datetime.strptime(line_items[1], "%m/%d/%Y %H:%M:%S")
                t = math.floor((date - date_min) / half_hour)
                if min_lng <= lng <= max_lng and min_lat <= lat <= max_lat and 0 <= t < total_len:
                    x, y = getxy(lng, lat)
                    count[t][0][y][x] = count[t][0][y][x] + 1

                lng = float(line_items[10])
                lat = float(line_items[9])
                date = datetime.datetime.strptime(line_items[2], "%m/%d/%Y %H:%M:%S")

                if date.month >= 6 and date.day >= 17:
                    # 留14天作为测试集
                    break

                t = math.floor((date - date_min) / half_hour)
                if min_lng <= lng <= max_lng and min_lat <= lat <= max_lng and 0 <= t < total_len:
                    x, y = getxy(lng, lat)
                    count[t][1][y][x] = count[t][1][y][x] + 1

            f.close()

        print(count.shape)
        total_count.append(count)

    total_count = np.concatenate(total_count, axis=0)
    print(total_count.shape)

    filename = 'C:\\Study and Work\\实验室\\PAPERS\\论文阅读第一期\\CitiBike\\output\\NYCBike_50x50_8slice_train.h5'
    fw = h5py.File(filename, 'w')
    fw['data'] = total_count
    fw.close()


if __name__ == '__main__':
    main()

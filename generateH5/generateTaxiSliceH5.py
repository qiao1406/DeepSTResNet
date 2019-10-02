import math
import h5py
import datetime
import numpy as np
import pandas as pd


max_lng = -73.92
min_lng = -74.02
max_lat = 40.8
min_lat = 40.67
grid = [50, 50]
grid_lng = (max_lng - min_lng) / grid[0]
grid_lat = (max_lat - min_lat) / grid[1]

half_hour = datetime.timedelta(minutes=30)
date_min = datetime.datetime(2016, 4, 1, 0, 0, 0) - datetime.timedelta(minutes=4)
date_max = datetime.datetime(2016, 6, 17, 0, 0, 0) - half_hour - datetime.timedelta(minutes=4)
total_len = (date_max - date_min) / half_hour


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


def analyze_slice(cnt):

    for month in ['2016-04', '2016-05', '2016-06']:

        for color in ['yellow', 'green']:

            path = '/mnt/windows-E/yjc/experiement/taxi/' + color + '_tripdata_' + month + '.csv'
            print('Now analyze:', path)
            csv_frame = pd.read_csv(path)

            for index, row in csv_frame.iterrows():

                if index % 10000 == 0:
                    print(index, '/', csv_frame.shape[0])

                lng = float(row[5])
                lat = float(row[6])

                date = datetime.datetime.strptime(row[1], "%Y-%m-%d %H:%M:%S")  # pick-up time
                t = math.floor((date - date_min) / half_hour)

                if t >= total_len:  # 超过最大时间了，要退出循环
                    break

                if min_lng <= lng <= max_lng and min_lat <= lat <= max_lat and 0 <= t < total_len:
                    x, y = getxy(lng, lat)
                    cnt[t][0][y][x] = cnt[t][0][y][x] + 1

                lng = float(row[7]) if color == 'green' else float(row[9])
                lat = float(row[8]) if color == 'green' else float(row[10])
                date = datetime.datetime.strptime(row[2], "%Y-%m-%d %H:%M:%S")  # drop-off time

                t = math.floor((date - date_min) / half_hour)
                if min_lng <= lng <= max_lng and min_lat <= lat <= max_lng and 0 <= t < total_len:
                    x, y = getxy(lng, lat)
                    cnt[t][1][y][x] = cnt[t][1][y][x] + 1


def main():

    total_count = []
    print(total_len)

    for shift in range(8):
        global date_min
        date_min += datetime.timedelta(minutes=4)
        print('slice from', date_min)

        count = np.zeros([int(total_len), 2, grid[0], grid[1]])

        analyze_slice(count)
        print(count.shape)
        total_count.append(count)

    total_count = np.concatenate(total_count, axis=0)
    print(total_count.shape)

    filename = '/mnt/windows-E/liwentao/DeepResNet/ST_NYC_Clustering/data/NewYork/taxi_50x50_8slice_train.h5'
    fw = h5py.File(filename, 'w')
    fw['data'] = total_count
    fw.close()


if __name__ == '__main__':
    main()

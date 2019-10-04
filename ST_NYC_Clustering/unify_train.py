import baselines.classical as bl
import dataprocess


bike_data_path = 'NYCBike_50x50_1slice.h5'
width = 50
height = 50


def main():
    pass


def history_average():

    recent_day = 7  # 使用的时间跨度

    data = dataprocess.load_train_data(91, bike_data_path)
    shape = data.shape
    data = data.reshape((shape[0], shape[1], width * height))

    channel = 0  # 选择预测 pick-up
    err = bl.history_average(data, width, height, channel, recent_day)
    print('Pick-up Err', err)

    channel = 1  # 选择预测 drop-off
    err = bl.history_average(data, width, height, channel, recent_day)
    print('Drop-off Err', err)


if __name__ == '__main__':
    history_average()

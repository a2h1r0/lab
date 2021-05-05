import sys
import numpy as np
import matplotlib.pyplot as plt
import csv
from natsort import natsorted
import glob
import os
os.chdir(os.path.dirname(__file__))


MODEL = 'TicWatch'  # 表示するスマートウォッチ
FILES = [['/Legion7/1st/20210504_142156_Pulse_TicWatchPro', 'Display A'],
         ['/OSOYOO/1st/20210504_190014_Pulse_TicWatchPro', 'Display B'],
         ['/body/20210505_142150_Pulse_TicWatchPro', 'body']]
START = 0  # 表示開始時刻
FINISH = 60  # 表示終了時刻

COLORS = ['red', 'blue', 'green']  # 描画色


def main():
    # グラフの準備
    plt.figure(figsize=(16, 9))
    plt.xlabel('Time [ms]', fontsize=18)
    plt.ylabel('Sensor Value', fontsize=18)
    plt.title('PPG Raw Data', fontsize=18)
    plt.tick_params(labelsize=18)

    # ディスプレイごとにデータを描画
    for data, color in zip(FILES, COLORS):
        with open('../generate_heart_rate/data/' + MODEL + data[0] + '.csv') as f:
            reader = csv.reader(f)
            next(reader)

            timestamps = []
            pulse = []
            for row in reader:
                timestamp = int(row[0])
                if timestamp < START * 1000:
                    # 開始時刻まで待機
                    continue
                elif FINISH * 1000 < timestamp:
                    # 終了時刻で終了
                    break

                # データの追加
                timestamps.append(timestamp)
                pulse.append(int(row[1]))

        plt.plot(timestamps, pulse, color, label=data[1])

    plt.legend(fontsize=18, loc='upper right')
    # plt.savefig('../figure/pulse_' + MODEL + '.eps',
    #             bbox_inches='tight', pad_inches=0)

    plt.show()


if __name__ == '__main__':
    main()

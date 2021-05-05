import sys
import numpy as np
import matplotlib.pyplot as plt
import csv
from natsort import natsorted
import glob
import os
os.chdir(os.path.dirname(__file__))


MODEL = 'TicWatch'  # 表示するスマートウォッチ
TARGET_RATES = [60, 65, 70, 75, 80, 85, 90, 95, 100]  # 取得した目標心拍数
DISPLAYS = [['Legion7', 'Display A'], ['OSOYOO', 'Display B']]  # 表示するディスプレイ
DIRS = ['1st', '2nd', '3rd']  # フォルダ分け
COLORS = ['red', 'blue']  # 描画色


def main():
    # グラフの準備
    plt.figure(figsize=(16, 9))
    plt.xlabel('Target Heart Rate', fontsize=18)
    plt.ylabel('Diff', fontsize=18)
    plt.title('Heart Rate', fontsize=18)
    plt.tick_params(labelsize=18)

    # ディスプレイごとにデータを描画
    for display, color in zip(DISPLAYS, COLORS):
        # 取得回数ごとの配列を作成
        diffs = [[] for i in range(len(DIRS))]
        for index, directory in enumerate(DIRS):
            files = natsorted(glob.glob('../generate_heart_rate/data/' +
                                        MODEL + '/' + display[0] + '/' + directory + '/*_HeartRate_*.csv'))

            if len(TARGET_RATES) != len(files):
                print('\nパラメータに誤りがあります。\n')
                sys.exit()

            # 心拍数ごとにデータを取得
            for target_rate, data in zip(TARGET_RATES, files):
                with open(data) as f:
                    reader = csv.reader(f)
                    next(reader)

                    values = []
                    for row in reader:
                        values.append(int(row[1]))

                    # 取得時間での平均値
                    average = round(sum(values) / len(values))
                    # 目標値からの差の計算
                    diffs[index].append(average - target_rate)

        # グラフの描画
        y = np.mean(diffs, axis=0)
        print(TARGET_RATES)
        print(y)
        plt.plot(TARGET_RATES, y, color, label=display[1])

    plt.legend(fontsize=18, loc='upper right')
    # plt.savefig('../figure/heartrate_' + MODEL + '.eps',
    #             bbox_inches='tight', pad_inches=0)

    plt.show()


if __name__ == '__main__':
    main()

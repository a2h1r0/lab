import matplotlib.pyplot as plt
import csv
from natsort import natsorted
import glob
import os
os.chdir(os.path.dirname(__file__))


MODEL = 'TicWatch'  # 表示するスマートウォッチ
TARGET_RATES = [70, 75, 80, 85, 90, 95, 100]  # 取得した目標心拍数
DISPLAYS = ['laptop', 'laptop']  # 表示するディスプレイ
COLORS = ['red', 'blue']  # 描画色


def main():
    # グラフの用意
    plt.figure(figsize=(16, 9))
    plt.xlabel('Target Heart Rate', fontsize=18)
    plt.ylabel('Diff', fontsize=18)
    plt.title(MODEL, fontsize=18)
    plt.tick_params(labelsize=18)

    # ディスプレイごとにデータを描画
    for index, (display, color) in enumerate(zip(DISPLAYS, COLORS)):
        files = natsorted(glob.glob('./data/' + MODEL +
                                    '/' + display + '/*_HeartRate_*.csv'))

        if len(TARGET_RATES) != len(files):
            print('\nパラメータに誤りがあります。\n')
            break

        # 心拍数ごとにデータを取得
        diffs = []
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
                diffs.append(average - target_rate)

        # グラフの描画
        plt.plot(TARGET_RATES, diffs, color, label=display)

    plt.legend(fontsize=18, loc='upper right')
    # plt.savefig('../figure/' + MODEL + '.png',
    #             bbox_inches='tight', pad_inches=0)

    plt.show()


if __name__ == '__main__':
    main()

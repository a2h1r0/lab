import sys
import numpy as np
import csv
from natsort import natsorted
import glob
import os
os.chdir(os.path.dirname(__file__))


MODEL = 'Series_5'  # 表示するスマートウォッチ
TARGET_RATES = [60, 65, 70, 75, 80, 85, 90, 95, 100]  # 取得した目標心拍数
DISPLAYS = [['Legion7', 'Display A'], ['ELECROW', 'Display B'],
            ['KeDei', 'Display C']]  # 表示するディスプレイ
DIRS = ['1st', '2nd', '3rd']  # フォルダ分け

INITIALIZATION_TIME = 30  # キャリブレーション時間
SENSING_TIME = 60  # データを使用する時間


def main():
    for display in DISPLAYS:
        # 取得回数ごとの配列を作成
        diffs = [[] for i in range(len(DIRS))]
        for index, directory in enumerate(DIRS):
            files = natsorted(glob.glob('../generate_heart_rate/data/AppleWatch/' +
                                        MODEL + '/' + display[0] + '/' + directory + '/*.csv'))

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
                        if INITIALIZATION_TIME * 1000 <= int(row[0]) and int(row[0]) <= (INITIALIZATION_TIME + SENSING_TIME) * 1000:
                            values.append(int(row[1]))

                    # INITIALIZATION_TIME以降のデータが存在しないときは最後のデータを使用
                    if len(values) == 0:
                        values.append(int(row[1]))

                    # 取得時間での平均値
                    get_rate = sum(values) / len(values)
                    # 目標値からの差の計算
                    diffs[index].append(get_rate - target_rate)

        # 3回の平均を計算
        averages = np.mean(diffs, axis=0)

        # 結果の表示
        print('\n--- ' + display[1] + ' ---\n')
        for target, diff in zip(TARGET_RATES, averages):
            print(str(target) + ': ' + str(round(diff, 3)))
        print('\nAverage Diff: ' + str(round(np.mean(averages), 3)))


if __name__ == '__main__':
    main()

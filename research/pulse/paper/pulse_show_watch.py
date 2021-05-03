import matplotlib.pyplot as plt
import csv
from natsort import natsorted
import glob
import os
os.chdir(os.path.dirname(__file__))


MODEL = 'TicWatch'  # 表示するスマートウォッチ
DISPLAYS = ['body', 'laptop']  # 表示するディスプレイ
FILES = ['20210503_175732_Pulse_TicWatchPro',
         '20210503_210144_Pulse_TicWatchPro']  # 表示するファイル
COLORS = ['red', 'blue']  # 描画色

START = 0  # 表示開始時刻
FINISH = 10  # 表示終了時刻

# グラフの用意
plt.figure(figsize=(16, 9))
plt.xlabel('Time [ms]', fontsize=18)
plt.ylabel('Sensor Value', fontsize=18)
plt.title(MODEL, fontsize=18)
plt.tick_params(labelsize=18)

# ディスプレイごとにデータを描画
for index, (display, filename, color) in enumerate(zip(DISPLAYS, FILES, COLORS)):
    data = '../make_heart_rate/data/' + MODEL + \
        '/' + display + '/' + filename + '.csv'

    # データの読み出し
    time = []
    pulse = []
    with open(data) as f:
        reader = csv.reader(f)
        next(reader)

        for row in reader:
            timestamp = int(row[0])
            if timestamp < START * 1000:
                # 開始時刻まで待機
                continue
            elif FINISH * 1000 < timestamp:
                # 終了時刻で終了
                break

            # データの追加
            time.append(timestamp)
            pulse.append(int(row[1]))

    # グラフの描画
    plt.plot(time, pulse, color, label=display)

plt.legend(fontsize=18, loc='upper right')
# plt.savefig('../figure/pulse_' + MODEL + '.eps',
#             bbox_inches='tight', pad_inches=0)

plt.show()

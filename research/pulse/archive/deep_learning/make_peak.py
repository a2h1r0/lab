import numpy as np
import serial
from time import sleep
import threading
from collections import deque
import socket
import pulse_module
import datetime
import csv
import random
import math
import matplotlib.pyplot as plt
from scipy import signal
import os
os.chdir(os.path.dirname(__file__))


USB_PORT = 'COM3'

SOCKET_ADDRESS = '192.168.11.2'  # Processingサーバのアドレス
SOCKET_PORT = 10000  # Processingサーバのポート


now = datetime.datetime.today()
time = now.strftime('%Y%m%d') + '_' + now.strftime('%H%M%S')
SAVE_DIR = './data/' + time + '/'

COLOR_DATA = SAVE_DIR + 'colors.csv'

TRAIN_DATA = '20210228_123306_raw'
START = 0
END = 3956
SAMPLE_SIZE = 74


def get_pulse():
    """脈波の取得
    脈波センサからデータを取得し，データ保存用キューを更新．
    """

    def make_display_data():
        """ランダム色データの生成

        Returns:
            list: 脈波配列
            list: 脈波から生成した色データ配列
        """

        pulse_data = []
        with open('./data/train/' + TRAIN_DATA + '.csv') as f:
            reader = csv.reader(f)

            # ヘッダーのスキップ
            next(reader)

            for row in reader:
                # データの追加
                pulse_data.append(float(row[1]))

        pulse_data = np.array(pulse_data[START:END])

        display_data = np.array(
            pulse_data / 1000 * 10 + 122, dtype=int)

        return pulse_data, list(display_data)

    raw, colors = make_display_data()
    generated = []

    with open('./data/sample.csv', 'w', newline='') as loss_file:
        # データの書き込み
        loss_writer = csv.writer(loss_file, delimiter=',')
        loss_writer.writerow(colors)

    # 色データの描画
    for color in colors:
        socket_client.send((str(color) + '\0').encode('UTF-8'))
        socket_client.recv(1)

        # 脈波値の受信
        data = ser.readline().rstrip().decode(encoding='UTF-8')

        if data.isdecimal() and len(data) <= 3:
            generated.append(int(data))
        else:
            # 異常値の場合
            continue

    raw = raw[len(raw) - SAMPLE_SIZE:]
    generated = generated[len(generated) - SAMPLE_SIZE:]

    # ピークの検出
    raw_peaks, _ = signal.find_peaks(raw, distance=50)
    generated_peaks, _ = signal.find_peaks(generated, distance=50)

    plt.figure(figsize=(16, 9))
    plt.plot(range(SAMPLE_SIZE), raw, 'red', label='Raw')
    plt.plot(range(SAMPLE_SIZE), generated, 'blue', label='Generated')
    for raw_peak, generated_peak in zip(raw_peaks, generated_peaks):
        plt.plot(raw_peak, raw[raw_peak], 'ro')
        plt.plot(generated_peak, generated[generated_peak], 'bo')
    plt.xlabel('Sample Num', fontsize=18)
    plt.ylabel('Pulse Value', fontsize=18)
    plt.tick_params(labelsize=18)
    plt.legend(fontsize=18, loc='upper right')
    # plt.savefig('../figures/' + save_figname,
    #             bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == '__main__':
    print('\n初期化中...')

    # シリアル通信（Arduino）の初期化
    ser = serial.Serial(USB_PORT, 14400)
    ser.reset_input_buffer()
    sleep(3)  # ポート準備に3秒待機**これがないとシリアル通信がうまく動かない**

    # ソケット通信（Processing）の初期化
    socket_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socket_client.connect((SOCKET_ADDRESS, SOCKET_PORT))

    # 脈波取得の開始
    get_pulse()

    # シリアル通信の終了
    ser.close()

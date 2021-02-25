import numpy as np
import torch
import torch.nn as nn
import torch.optim as optimizers
import serial
from time import sleep
import threading
from collections import deque
import socket
from model import Pix2Pix
import pulse_module
import datetime
import csv
import random
import math
import os
os.chdir(os.path.dirname(__file__))


USB_PORT = 'COM3'

SOCKET_ADDRESS = '192.168.11.2'  # Processingサーバのアドレス
SOCKET_PORT = 10000  # Processingサーバのポート


SAMPLE_SIZE = 1000  # サンプルサイズ
EPOCH_NUM = 30000  # 学習サイクル数
KERNEL_SIZE = 13  # カーネルサイズ（奇数のみ）
LAMBDA = 10.0  # 損失の比率パラメータ

FILE_EPOCH_NUM = 30000  # 1ファイルに保存するエポック数

now = datetime.datetime.today()
time = now.strftime('%Y%m%d') + '_' + now.strftime('%H%M%S')
SAVE_DIR = './data/' + time + '/'

COLOR_DATA = SAVE_DIR + 'colors.csv'
LOSS_DATA = SAVE_DIR + 'loss.csv'

TRAIN_DATAS = ['20210207_121945_raw', '20210207_122512_raw',
               '20210207_123029_raw', '20210207_123615_raw',
               '20210207_154330_raw']


def make_display_data():
    """ランダム色データの生成

    Args:
        radian (int): ラジアン周期
    Returns:
        int: 色データ
    """

    #*** 学習ファイルデータ用変数 ***#
    global train_data

    pulse_data = np.array(
        train_data[random.randrange(0, len(train_data) - 1)])
    display_data = pulse_data / max(pulse_data) * 5 + 128

    return list(display_data)


if __name__ == '__main__':
    #*** グローバル：学習ファイルデータ用変数 ***#
    train_data = []
    for data in TRAIN_DATAS:
        with open('./data/train/' + data + '.csv') as f:
            reader = csv.reader(f)

            # ヘッダーのスキップ
            next(reader)

            read_data = []
            for row in reader:
                # データの追加
                read_data.append(float(row[1]))
        train_data.append(read_data)

    # ソケット通信（Processing）の初期化
    socket_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socket_client.connect((SOCKET_ADDRESS, SOCKET_PORT))

    display_data = []
    while True:
        try:
            # 色データの作成
            if len(display_data) < SAMPLE_SIZE:
                display_data = make_display_data()

            # 色データの描画
            color = display_data.pop(0)
            socket_client.send((str(color) + '\0').encode('UTF-8'))
            socket_client.recv(1)

        except KeyboardInterrupt:
            break

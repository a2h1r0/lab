import numpy as np
import csv
import socket
import time
import os
os.chdir(os.path.dirname(__file__))


SOCKET_ADDRESS = '192.168.11.2'  # Processingサーバのアドレス
SOCKET_PORT = 10000  # Processingサーバのポート


TRAIN_DATA = '20210228_124511_raw'
PROCESS_TIME = 240


def light():
    """点灯"""

    def make_display_data(heart_rate):
        """ランダム色データの生成

        Args:
            heart_rate (int): 再現する心拍数
        Returns:
            list: 色データ配列
        """

        wave = []
        timestamps = []
        pulse_data = []
        with open('./data/sample.csv') as f:
            reader = csv.reader(f)

            for row in reader:
                single_wave = row

            single_wave = np.array(single_wave, dtype=int)
            single_wave = (single_wave / min(single_wave)
                           * 10) + min(single_wave)
            single_wave.astype(int)

        wave = []
        for i in range(heart_rate):
            wave.extend(single_wave)

        colors = np.array(wave, dtype=int)

        return colors

    heart_rate = input('心拍数は？ > ')

    # 色データの作成
    colors = make_display_data(int(heart_rate))
    # 点灯時間の計算
    lighting_time = 60 / len(colors)

    # 開始時間の取得
    start = time.time()

    # 色データの描画
    while True:
        for color in colors:
            process = time.time() - start

            if process > PROCESS_TIME:
                break

            socket_client.send((str(color) + '\0').encode('UTF-8'))
            socket_client.recv(1)

            # サンプリングレートと点灯時間を合わせる
            next_time = process + lighting_time
            while True:
                process = time.time() - start
                if next_time - process <= 0:
                    break

        if process > PROCESS_TIME:
            break


if __name__ == '__main__':
    # ソケット通信（Processing）の初期化
    socket_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socket_client.connect((SOCKET_ADDRESS, SOCKET_PORT))

    # 点灯開始
    light()

import numpy as np
import csv
import datetime
import socket
import time
import os
os.chdir(os.path.dirname(__file__))


PROCESS_TIME = 240  # 実行時間
LOG_FILE = './data/run.log'  # ログファイル


SOCKET_ADDRESS = '192.168.11.2'  # Processingサーバのアドレス
SOCKET_PORT = 10000  # Processingサーバのポート


def light(heart_rate):
    """点灯

        Args:
            heart_rate (int): 再現する心拍数
    """

    def make_display_data(heart_rate):
        """色データの生成

        Args:
            heart_rate (int): 再現する心拍数
        Returns:
            list: 色データ配列
            list: 点灯時間
        """

        # sin波（0 ~ 1）の生成
        sin = np.sin(np.linspace(np.pi, 2 * np.pi, 10)) + 1
        ones = np.ones(10)

        # グレースケールへ変換
        single_wave = np.append(ones, sin) * 30 + 225

        wave = []
        for i in range(heart_rate):
            wave.extend(single_wave)

        colors = np.array(wave, dtype=int)

        # 点灯時間の計算
        lighting_time = 60 / len(colors)

        return colors, lighting_time

    # 色データの作成
    colors, lighting_time = make_display_data(int(heart_rate))

    # 描画開始時間の取得
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

    # 心拍数の設定
    heart_rate = input('\n\n心拍数は？ > ')

    # プログラム実行日時と実行時間，心拍数を記録
    with open(LOG_FILE, 'a', newline='') as log_file:
        log_writer = csv.writer(log_file, delimiter=',')
        now = datetime.datetime.today()
        date_time = now.strftime('%Y/%m/%d') + ' ' + now.strftime('%H:%M:%S')
        log_writer.writerow([date_time, PROCESS_TIME, heart_rate])

    print('\n描画中．．．')

    # 点灯開始
    light(heart_rate)

    print('\n----- 描画終了 -----\n')

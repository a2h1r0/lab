import numpy as np
import csv
import socket
import time
import os
os.chdir(os.path.dirname(__file__))


SOCKET_ADDRESS = '192.168.11.2'  # Processingサーバのアドレス
SOCKET_PORT = 10000  # Processingサーバのポート


TRAIN_DATA = '20210228_124511_raw'
LIGHTING_TIME = 10


def light():
    """点灯"""

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

        pulse_data = np.array(pulse_data)

        display_data = np.array(
            pulse_data / max(pulse_data) * 5 + 122, dtype=int)

        return pulse_data, list(display_data)

    raw, colors = make_display_data()
    generated = []

    # 開始時間の取得
    start = time.time()

    # 色データの描画
    for color in colors:
        if (time.time() - start) > LIGHTING_TIME:
            break

        socket_client.send((str(color) + '\0').encode('UTF-8'))
        socket_client.recv(1)


if __name__ == '__main__':
    # ソケット通信（Processing）の初期化
    socket_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socket_client.connect((SOCKET_ADDRESS, SOCKET_PORT))

    print('\n点灯中...\n')
    light()

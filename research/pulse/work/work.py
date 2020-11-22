import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import serial
from time import sleep
import threading
from collections import deque
import socket


SAMPLE_SIZE = 64  # サンプルサイズ（学習して再現する脈波の長さ）

TESTDATA_SIZE = 0.3  # テストデータの割合

# EPOCH_NUM = 1000  # 学習サイクル数
EPOCH_NUM = 1  # 学習サイクル数

WINDOW_SIZE = 32  # ウィンドウサイズ
STEP_SIZE = 1  # ステップ幅
BATCH_SIZE = WINDOW_SIZE  # バッチサイズ

VECTOR_SIZE = 1  # 扱うベクトルのサイズ（脈波は1次元）

INPUT_DIMENSION = 1  # LSTMの入力次元数（脈波の時系列データは各時刻で1次元）
HIDDEN_SIZE = 24  # LSTMの隠れ層
OUTPUT_DIMENSION = SAMPLE_SIZE  # LSTMの出力次元数（SAMPLE_SIZE個の色データ）

USB_PORT = 'COM3'

SOCKET_ADDRESS = '192.168.11.2'  # Processingサーバのアドレス
SOCKET_PORT = 10000  # Processingサーバのポート


class LSTM(nn.Module):
    """
    LSTMモデル
    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        Args:
            input_size (int): 入力次元数
            hidden_size (int): 隠れ層サイズ
            output_size (int): 出力サイズ
        """

        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(
            input_size=input_size, hidden_size=hidden_size)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, input):
        """
        Args:
            input (:obj:`Tensor`): 学習データ

        Returns:
            :obj:`Numpy`: 予測結果の色データ
        """

        _, lstm_out = self.lstm(input)
        linear_out = self.fc(lstm_out[0].view(-1, self.hidden_size))
        out = torch.sigmoid(linear_out)

        # 色データの生成
        color_data = self.convert_to_color_data(out)

        # 予測結果の色データを出力
        return color_data

    def convert_to_color_data(self, out):
        """色データへの変換

        予測結果から色データを生成する．

        Args:
            out (:obj:`Tensor`): 予測結果

        Returns:
            :obj:`Numpy`: 予測結果の色データ
        """

        # 色の最大値は16進数FFFFFF（色コード）
        COLOR_MAX = 16777215

        # Tensorから1次元のNumpyへ
        out = out.detach().cpu().numpy().reshape(-1)

        # 出力の最大値を色の最大値に合わせる（整数）
        converted_data = np.array(out * COLOR_MAX, dtype=int)

        return converted_data.astype('str')


def get_pulse():
    """脈波の取得

    脈波センサからデータを取得し，データ保存用キューを更新．
    """

    #*** データ送信用変数 ***#
    global send_to_display_data
    #*** ディスプレイ点灯時間用変数 ***#
    global display_lighting_time
    #*** 擬似脈波取得開始時間用変数 ***#
    global pseudo_pulse_get_start_time

    # 脈波値の受信
    read_data = ser.readline().rstrip().decode(encoding='UTF-8')
    # data[0]: micros, data[1]: raw_pulse, data[2]: pseudo_pulse
    data = read_data.split(",")
    print(data)

    # 正常値が受信できていることを確認
    if len(data) == 3 and data[0].isdecimal() and data[1].isdecimal() and data[2].isdecimal():
        timestamp = float(data[0])/1000
        # センサ値取得時間用キューの更新（単位はミリ秒で保存）
        pulse_get_timestamps.append(timestamp)
        # 生脈波用キューの更新
        raw_pulse_values.append(data[1])

        # 送信するデータが存在する場合（ディスプレイ点灯開始時）または，データ取得中の場合
        if (send_to_display_data is not None) or (pseudo_pulse_get_start_time is not None):
            # ディスプレイ点灯開始時に時刻を保存
            if pseudo_pulse_get_start_time is None:
                # 脈波の取得開始時刻
                pseudo_pulse_get_start_time = timestamp

            # 点灯時間（学習データと同じ時間）だけ取得
            # 現在時刻が(取得開始時刻 + 点灯時間)より大きいかつ，サンプル数が学習データと同じだけ集まったら取得終了
            if (timestamp > (pseudo_pulse_get_start_time + display_lighting_time)) and (len(pseudo_pulse_values) == SAMPLE_SIZE):
                # 脈波の取得開始時刻の初期化
                pseudo_pulse_get_start_time = None

            # 取得時間内
            else:
                # 擬似脈波用キューの更新
                pseudo_pulse_values.append(data[2])


def draw_display():
    """ディスプレイの描画

    色データを送信し，ディスプレイに描画．

    Args:
        color (int): 色データ

    Returns:
        pulse_value (int): 脈波値
    """

    #*** データ送信用変数 ***#
    global send_to_display_data
    #*** ディスプレイ点灯時間用変数 ***#
    global display_lighting_time

    while True:
        # 送信するデータが存在する場合
        if send_to_display_data is not None:
            # 1サンプルあたりの点灯時間の取得（全サンプルでの点灯時間 ÷ サンプル長）
            color_lighting_time = str(
                display_lighting_time / len(send_to_display_data))

            # 1サンプルずつ送信
            for color_data in send_to_display_data:
                # 終端文字の追加
                color_data += '\0'

                # 送信のためにデータを整形（点灯時間,色データ\0）
                data = color_lighting_time + ',' + color_data

                # 色データの送信
                socket_client.send(data.encode('UTF-8'))

            # 送信データの初期化（完了通知）
            send_to_display_data = None


def train():
    """
    学習
    """

    #*** データ送信用変数 ***#
    global send_to_display_data
    #*** ディスプレイ点灯時間用変数 ***#
    global display_lighting_time

    # モデルの定義
    model = LSTM(input_size=INPUT_DIMENSION,
                 hidden_size=HIDDEN_SIZE, output_size=OUTPUT_DIMENSION)
    model.to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters())

    model.train()
    optimizer.zero_grad()

    #--- 学習サイクル ---#
    # for epoch in range(EPOCH_NUM):
    # print('EPOCH: ' + str(epoch))

    # データが貯まるまで待機
    while True:
        # サンプルがSAMPLE_SIZE個存在している場合
        if len(raw_pulse_values) == SAMPLE_SIZE:
            break

    #--- データセットの作成 ---#
    # 学習に使用するデータの取得
    train_pulse_values = np.array(raw_pulse_values, dtype=int)

    # 全サンプルでの点灯時間の取得（最終サンプルのタイムスタンプ - 開始サンプルのタイムスタンプ）
    display_lighting_time = pulse_get_timestamps[-1] - \
        pulse_get_timestamps[0]

    # LSTM入力形式に変換
    train_pulse_values = torch.tensor(
        train_pulse_values, dtype=torch.float, device=device).view(-1, 1, 1)

    #--- 学習 ---#
    # 予測値（色データ）の取得
    colors = model(train_pulse_values)

    # ディスプレイ送信用データの作成
    send_to_display_data = colors

    # 処理が完了するまで待機
    while True:
        # データの送信が完了しているかつ，脈波の取得が完了している場合
        if (send_to_display_data is None) and (pseudo_pulse_get_start_time is None):
            # ディスプレイ点灯時間の初期化
            display_lighting_time = None
            break

    print(train_pulse_values)
    print(pseudo_pulse_values)

    # # 予測結果を1件ずつ処理
    # pulse_values = []
    # for color in colors:
    #     # ディスプレイの描画と脈波の取得
    #     pulse_value = send_color_and_get_pulse(color)
    #     pulse_values.append(pulse_value)

    # print(pulse_values)


def main():
    # 学習スレッドの開始
    train_thread = threading.Thread(target=train)
    train_thread.setDaemon(True)
    train_thread.start()
    # ディスプレイ制御スレッドの開始
    draw_display_thread = threading.Thread(target=draw_display)
    draw_display_thread.setDaemon(True)
    # draw_display_thread.start()

    while True:
        try:
            # 脈波の更新
            get_pulse()
        except KeyboardInterrupt:
            break


if __name__ == '__main__':
    print("\n初期化中...")

    # センサ値取得時間用キュー
    pulse_get_timestamps = deque(maxlen=SAMPLE_SIZE)
    # 生脈波用キュー
    raw_pulse_values = deque(maxlen=SAMPLE_SIZE)
    # 擬似脈波用キュー
    pseudo_pulse_values = deque(maxlen=SAMPLE_SIZE)

    #*** グローバル：データ送信用変数（画面点灯の制御） ***#
    send_to_display_data = None
    #*** グローバル：擬似脈波取得開始時間用変数（擬似脈波取得の制御） ***#
    pseudo_pulse_get_start_time = None
    #*** グローバル：ディスプレイ点灯時間用変数（画面点灯時間，擬似脈波取得時間の制御） ***#
    display_lighting_time = None

    # PyTorchの初期化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1)

    # シリアル通信（Arduino）の初期化
    ser = serial.Serial(USB_PORT, 115200)
    ser.reset_input_buffer()
    sleep(3)  # ポート準備に3秒待機**これがないとシリアル通信がうまく動かない**

    # ソケット通信（Processing）の初期化
    socket_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socket_client.connect((SOCKET_ADDRESS, SOCKET_PORT))

    # メイン処理
    main()

    # シリアル通信の終了
    ser.close()

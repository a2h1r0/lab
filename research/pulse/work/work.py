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
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from soft_dtw import SoftDTW
import datetime
import csv
import os
os.chdir(os.path.dirname(__file__))


SAMPLE_SIZE = 256  # サンプルサイズ（学習して再現する脈波の長さ）

TESTDATA_SIZE = 0.3  # テストデータの割合

EPOCH_NUM = 100  # 学習サイクル数

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

now = datetime.datetime.today()
time = now.strftime("%Y%m%d") + "_" + now.strftime("%H%M%S")
SAVEFILE_RAW = time + "_raw.csv"
SAVEFILE_PSEUDO = time + "_pseudo.csv"


class GAN(nn.Module):
    """
    LSTMモデル
    """

    def __init__(self, device='cpu'):
        """
        Args:
            input_size (int): 入力次元数
            hidden_size (int): 隠れ層サイズ
            output_size (int): 出力サイズ
        """

        super().__init__()
        self.device = device
        self.G = Generator(device=device)
        self.D = Discriminator(device=device)

    def forward(self, x):
        """
        Args:
            input (:obj:`Tensor`): 学習データ

        Returns:
            :obj:`Numpy`: 予測結果の色データ
        """

        x = self.G(x)
        y = self.D(x)

        return y


class Discriminator(nn.Module):
    """
    LSTMモデル
    """

    def __init__(self, device='cpu'):
        """
        Args:
            input (:obj:`Tensor`): 学習データ

        Returns:
            :obj:`Numpy`: 予測結果の色データ
        """

        super().__init__()
        self.device = device
        self.conv1 = nn.Conv2d(1, 128,
                               kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.relu1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(128, 256,
                               kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.LeakyReLU(0.2)
        self.fc = nn.Linear(256*7*7, 1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.relu3 = nn.LeakyReLU(0.2)
        self.out = nn.Linear(1024, 1)

    def forward(self, x):
        """
        Args:
            input (:obj:`Tensor`): 学習データ

        Returns:
            :obj:`Numpy`: 予測結果の色データ
        """

        h = self.conv1(x)
        h = self.relu1(h)
        h = self.conv2(h)
        h = self.bn2(h)
        h = self.relu2(h)
        h = h.view(-1, 256*7*7)
        h = self.fc(h)
        h = self.bn3(h)
        h = self.relu3(h)
        h = self.out(h)
        y = torch.sigmoid(h)

        return y


class Generator(nn.Module):
    """
    LSTMモデル
    """

    def __init__(self, input_dim=100, device='cpu'):
        """
        Args:
            input_size (int): 入力次元数
            hidden_size (int): 隠れ層サイズ
            output_size (int): 出力サイズ
        """

        super().__init__()
        self.device = device
        self.linear = nn.Linear(input_dim, 256*14*14)
        self.bn1 = nn.BatchNorm1d(256*14*14)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(256, 128,
                               kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(128, 64,
                               kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 1, kernel_size=(1, 1))

    def forward(self, x):
        """
        Args:
            input_size (int): 入力次元数
            hidden_size (int): 隠れ層サイズ
            output_size (int): 出力サイズ
        """

        h = self.linear(x)
        h = self.bn1(h)
        h = self.relu1(h)
        h = h.view(-1, 256, 14, 14)
        h = F.interpolate(h, size=(28, 28))
        h = self.conv1(h)
        h = self.bn2(h)
        h = self.relu2(h)
        h = self.conv2(h)
        h = self.bn3(h)
        h = self.relu3(h)
        h = self.conv3(h)
        y = torch.sigmoid(h)

        return y


def get_pulse():
    """脈波の取得

    脈波センサからデータを取得し，データ保存用キューを更新．
    """

    #*** 学習生脈波用変数 ***#
    global train_raw_pulse
    #*** データ送信用変数 ***#
    global send_to_display_data
    #*** ディスプレイ点灯時間用変数 ***#
    global display_lighting_time
    #*** 学習擬似脈波用変数 ***#
    global pseudo_pulse

    #*** 処理終了通知用変数 ***#
    global finish

    # データ書き込みファイルのオープン
    raw_file = open(SAVEFILE_RAW, 'x', newline='')
    raw_writer = csv.writer(raw_file, delimiter=',')
    raw_writer.writerow(["time", "pulse"])
    pseudo_file = open(SAVEFILE_PSEUDO, 'x', newline='')
    pseudo_writer = csv.writer(pseudo_file, delimiter=',')
    pseudo_writer.writerow(["time", "pulse"])

    # 脈波の取得開始時刻の初期化
    pseudo_pulse_get_start_time = None

    # 終了フラグが立つまで脈波を取得し続ける
    while not finish:
        try:
            # 脈波値の受信
            read_data = ser.readline().rstrip().decode(encoding='UTF-8')
            # data[0]: micros, data[1]: raw_pulse, data[2]: pseudo_pulse
            data = read_data.split(",")
            # print(data)

            # 正常値が受信できていることを確認
            if len(data) == 3 and data[0].isdecimal() and data[1].isdecimal() and data[2].isdecimal():
                timestamp = float(data[0])/1000

                #--- データの保存 ---#
                raw_writer.writerow([timestamp, int(data[1])])
                pseudo_writer.writerow([timestamp, int(data[2])])

                # センサ値取得時間用キューの更新（単位はミリ秒で保存）
                pulse_get_timestamps.append(timestamp)
                # 生脈波用キューの更新
                raw_pulse_values.append(int(data[1]))

                #--- データセットの作成 ---#
                # サンプルがSAMPLE_SIZE個貯まるまで待機
                if (len(raw_pulse_values) == SAMPLE_SIZE) and (train_raw_pulse is None):
                    # 全サンプルでの点灯時間の取得（最終サンプルのタイムスタンプ - 開始サンプルのタイムスタンプ）
                    display_lighting_time = pulse_get_timestamps[-1] - \
                        pulse_get_timestamps[0]
                    # 学習に使用するデータの取得
                    train_raw_pulse = raw_pulse_values
                    # print('生脈波取得完了')

                # ディスプレイ点灯開始時に時刻を保存
                if (send_to_display_data is not None) and (pseudo_pulse_get_start_time is None):
                    # 脈波の取得開始時刻（データ取得中状態）
                    pseudo_pulse_get_start_time = timestamp
                    # 取得開始時刻の書き込み
                    pseudo_writer.writerow([timestamp, 'start'])

                # データ取得中かつ，擬似脈波受付可能状態の場合
                if (pseudo_pulse_get_start_time is not None) and (pseudo_pulse is None):

                    # 点灯時間（学習データと同じ時間）だけ取得
                    # 現在時刻が(取得開始時刻 + 点灯時間)より大きいかつ，サンプル数が学習データと同じだけ集まったら取得終了
                    if (timestamp > (pseudo_pulse_get_start_time + display_lighting_time)) and (len(pseudo_pulse_values) == SAMPLE_SIZE):
                        # ディスプレイ点灯時間の初期化
                        display_lighting_time = None
                        # 脈波の取得開始時刻の初期化
                        pseudo_pulse_get_start_time = None
                        # 学習用に擬似脈波をコピー
                        pseudo_pulse = pseudo_pulse_values

                        # 取得完了時刻の書き込み
                        pseudo_writer.writerow([timestamp, 'finish'])
                        # print('擬似脈波取得完了')

                    # 取得時間内
                    else:
                        # 擬似脈波用キューの更新
                        pseudo_pulse_values.append(int(data[2]))

        except KeyboardInterrupt:
            break

    # データ書き込みファイルのクローズ
    raw_file.close()
    pseudo_file.close()


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

        # 描画通知の待機
        socket_client.recv(1)
        # print(socket_client.recv(1))

    # 送信データの初期化（完了通知）
    send_to_display_data = None


def main():
    """
    メイン処理
    """

    #*** 学習生脈波用変数 ***#
    global train_raw_pulse
    #*** データ送信用変数 ***#
    global send_to_display_data
    #*** ディスプレイ点灯時間用変数 ***#
    global display_lighting_time
    #*** 学習擬似脈波用変数 ***#
    global pseudo_pulse

    #*** 処理終了通知用変数 ***#
    global finish

    def get_pesudo_pulse(colors):
        """擬似脈波の取得

        Args:
            colors (int): 色データ
        """

        # 学習擬似脈波の初期化
        pseudo_pulse = None

        # ディスプレイ送信用データの作成（Tensorから1次元の整数，文字列のNumpyへ）
        send_to_display_data = np.array(
            colors.detach().cpu().numpy().reshape(-1), dtype=int).astype('str')

        # 描画開始
        # print('描画開始')
        draw_display()
        # print('描画終了')

        # 擬似脈波の取得が完了するまで待機
        while pseudo_pulse is None:
            # 1μsの遅延**これを入れないと回りすぎてセンサデータ取得の動作が遅くなる**
            sleep(0.000001)
            continue

    '''モデルの構築'''
    model = GAN(device=device).to(device)

    def convert_to_image(pulse):
        """脈波値を画像に変換

        Args:
            pulse (int): 脈波値

        Returns:
            :obj:`Tensor`: 画像データ
        """

        # todo: 二次元フーリエ変換
        return torch.empty(batch_size, 100).uniform_(0, 1).to(device)

    '''モデルの訓練'''
    criterion = nn.BCELoss()
    optimizer_D = optimizers.Adam(model.D.parameters(), lr=0.0002)
    optimizer_G = optimizers.Adam(model.G.parameters(), lr=0.0002)

    def compute_loss(preds, label):
        """損失の計算

        Args:
            preds (:obj:`Tensor`): 予測結果
            label (:obj:`Tensor`): 正解ラベル

        Returns:
            :obj:`Tensor`: 損失
        """

        return criterion(preds, label)

    def train_step(raw_pulse):
        model.D.train()
        model.G.train()

        # ---------------------
        #  識別器の学習
        # ---------------------
        #-- 本物データ --#
        # 生波形に対する予測
        raw_pulse_img = convert_to_image(raw_pulse)
        preds = model.D(raw_pulse_img).squeeze()
        label = torch.ones(1).float().to(device)
        loss_D_real = compute_loss(preds, label)

        #-- 偽物（擬似）データ --#
        # 生波形から色データを生成
        colors = model.G(raw_pulse_img)
        get_pesudo_pulse(colors)
        # 擬似脈波に対する予測
        pseudo_pulse_img = convert_to_image(pseudo_pulse)
        preds = model.D(pseudo_pulse_img.detach()).squeeze()
        label = torch.zeros(1).float().to(device)
        loss_D_fake = compute_loss(preds, label)

        #-- 学習 --#
        loss_D = loss_D_real + loss_D_fake
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # ---------------------
        #  生成器の学習
        # ---------------------
        # 生波形から色データを生成
        colors = model.G(raw_pulse_img)
        get_pesudo_pulse(colors)
        # 擬似脈波に対する予測
        pseudo_pulse_img = convert_to_image(pseudo_pulse)
        preds = model.D(pseudo_pulse_img).squeeze()
        label = torch.ones(1).float().to(device)  # 偽物画像のラベルを「本物画像(1)」とする
        loss_G = compute_loss(preds, label)

        #-- 学習 --#
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        return loss_D, loss_G

    '''学習サイクル'''
    for epoch in range(EPOCH_NUM):
        # 生脈波の取得が完了するまで待機
        while train_raw_pulse is None:
            # 1μsの遅延**これを入れないと回りすぎてセンサデータ取得の動作が遅くなる**
            sleep(0.000001)
            continue

        # 学習
        loss_D, loss_G = train_step(train_raw_pulse)

        print('Epoch: {}, D Cost: {:.3f}, G Cost: {:.3f}'.format(
            epoch+1, loss_D.item(), loss_G.item()))

        # 学習完了通知（次の脈波を取得）
        train_raw_pulse = None
        pseudo_pulse = None

    # 処理終了
    finish = True


if __name__ == '__main__':
    print("\n初期化中...")

    # センサ値取得時間用キュー
    pulse_get_timestamps = deque(maxlen=SAMPLE_SIZE)
    # 生脈波用キュー
    raw_pulse_values = deque(maxlen=SAMPLE_SIZE)
    # 擬似脈波用キュー
    pseudo_pulse_values = deque(maxlen=SAMPLE_SIZE)

    #*** グローバル：処理終了通知用変数（センサデータ取得終了の制御） ***#
    finish = False

    #*** グローバル：学習生脈波用変数 ***#
    train_raw_pulse = None
    #*** グローバル：学習擬似脈波用変数 ***#
    pseudo_pulse = None
    #*** グローバル：データ送信用変数（画面点灯の制御） ***#
    send_to_display_data = None
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

    # 学習スレッドの開始
    train_thread = threading.Thread(target=main)
    train_thread.setDaemon(True)
    train_thread.start()

    # 脈波取得の開始
    get_pulse()

    # シリアル通信の終了
    ser.close()

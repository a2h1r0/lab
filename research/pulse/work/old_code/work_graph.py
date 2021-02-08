import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import serial
from time import sleep
import threading
from collections import deque
import socket
from scipy.spatial.distance import euclidean
from soft_dtw import SoftDTW
import datetime
import csv
import random
import os
os.chdir(os.path.dirname(__file__))


USB_PORT = 'COM3'

SOCKET_ADDRESS = '192.168.11.2'  # Processingサーバのアドレス
SOCKET_PORT = 10000  # Processingサーバのポート


SAMPLE_SIZE = 2048  # サンプルサイズ（学習して再現する脈波の長さ）

EPOCH_NUM = 10000  # 学習サイクル数

WINDOW_SIZE = 32  # ウィンドウサイズ
STEP_SIZE = 1  # ステップ幅
BATCH_SIZE = WINDOW_SIZE  # バッチサイズ
MAP_SIZE = 8  # CNNのマップサイズ
KERNEL_SIZE = 10  # カーネルサイズ

VECTOR_SIZE = 1  # 扱うベクトルのサイズ（脈波は1次元）

INPUT_DIMENSION = 1  # LSTMの入力次元数（脈波の時系列データは各時刻で1次元）
HIDDEN_SIZE = 24  # LSTMの隠れ層
OUTPUT_DIMENSION = SAMPLE_SIZE  # LSTMの出力次元数（SAMPLE_SIZE個の色データ）

now = datetime.datetime.today()
time = now.strftime("%Y%m%d") + "_" + now.strftime("%H%M%S")
SAVEFILE_RAW = time + "_raw.csv"
SAVEFILE_GENERATED = time + "_generated.csv"
SAVEFILE_GENERATED2 = time + "_autoencoder.csv"

TRAIN_DATA = '20201202_154312_raw.csv'
LOSS_DATA = time + '_loss.csv'


class Pix2Pix(nn.Module):
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

        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=8, kernel_size=KERNEL_SIZE)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(
            in_channels=8, out_channels=16, kernel_size=KERNEL_SIZE)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv1d(
            in_channels=16, out_channels=32, kernel_size=KERNEL_SIZE)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=KERNEL_SIZE)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=KERNEL_SIZE)
        self.relu5 = nn.ReLU(inplace=True)

        self.conv6 = nn.Conv1d(
            in_channels=128, out_channels=256, kernel_size=KERNEL_SIZE)
        self.relu6 = nn.ReLU(inplace=True)

        self.conv7 = nn.Conv1d(
            in_channels=256, out_channels=1, kernel_size=1)

    def forward(self, input):
        """
        Args:
            input (:obj:`Tensor`): 学習データ

        Returns:
            :obj:`Tensor`: 識別結果
        """

        conv1_out = self.conv1(input)
        relu1_out = self.relu1(conv1_out)

        conv2_out = self.conv2(relu1_out)
        relu2_out = self.relu2(conv2_out)

        conv3_out = self.conv3(relu2_out)
        relu3_out = self.relu3(conv3_out)

        conv4_out = self.conv4(relu3_out)
        relu4_out = self.relu4(conv4_out)

        conv5_out = self.conv5(relu4_out)
        relu5_out = self.relu5(conv5_out)

        conv6_out = self.conv6(relu5_out)
        relu6_out = self.relu6(conv6_out)

        out = self.conv7(relu6_out)

        return out


class Generator(nn.Module):
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

        ###--- Encoder ---###
        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=8, kernel_size=KERNEL_SIZE)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(
            in_channels=8, out_channels=16, kernel_size=KERNEL_SIZE)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv1d(
            in_channels=16, out_channels=32, kernel_size=KERNEL_SIZE)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=KERNEL_SIZE)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=KERNEL_SIZE)
        self.relu5 = nn.ReLU(inplace=True)

        ###--- Decoder ---###
        self.conv6 = nn.ConvTranspose1d(
            in_channels=128, out_channels=64, kernel_size=KERNEL_SIZE)
        self.relu6 = nn.ReLU(inplace=True)

        # Skip Connection (conv4)
        self.conv7 = nn.ConvTranspose1d(
            in_channels=64 * 2, out_channels=32, kernel_size=KERNEL_SIZE)
        self.relu7 = nn.ReLU(inplace=True)

        # Skip Connection (conv3)
        self.conv8 = nn.ConvTranspose1d(
            in_channels=32 * 2, out_channels=16, kernel_size=KERNEL_SIZE)
        self.relu8 = nn.ReLU(inplace=True)

        # Skip Connection (conv2)
        self.conv9 = nn.ConvTranspose1d(
            in_channels=16 * 2, out_channels=8, kernel_size=KERNEL_SIZE)
        self.relu9 = nn.ReLU(inplace=True)

        # Skip Connection (conv1)
        self.conv10 = nn.ConvTranspose1d(
            in_channels=8 * 2, out_channels=1, kernel_size=KERNEL_SIZE)
        self.relu10 = nn.ReLU(inplace=True)

    def forward(self, input):
        """
        Args:
            input_size (int): 入力次元数
            hidden_size (int): 隠れ層サイズ
            output_size (int): 出力サイズ
        """

        ###--- Encoder ---###
        conv1_out = self.conv1(input)
        x1 = self.relu1(conv1_out)

        conv2_out = self.conv2(x1)
        x2 = self.relu2(conv2_out)

        conv3_out = self.conv3(x2)
        x3 = self.relu3(conv3_out)

        conv4_out = self.conv4(x3)
        x4 = self.relu4(conv4_out)

        conv5_out = self.conv5(x4)
        x5 = self.relu5(conv5_out)

        ###--- Decoder ---###
        conv6_out = self.conv6(x5)
        relu6_out = self.relu6(conv6_out)

        # Skip Connection (conv4)
        conv7_out = self.conv7(torch.cat([relu6_out, x4], dim=1))
        relu7_out = self.relu7(conv7_out)

        # Skip Connection (conv3)
        conv8_out = self.conv8(torch.cat([relu7_out, x3], dim=1))
        relu8_out = self.relu8(conv8_out)

        # Skip Connection (conv2)
        conv9_out = self.conv9(torch.cat([relu8_out, x2], dim=1))
        relu9_out = self.relu9(conv9_out)

        # Skip Connection (conv1)
        conv10_out = self.conv10(torch.cat([relu9_out, x1], dim=1))
        out = self.relu10(conv10_out)

        return out


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
    global generated_pulse

    #*** 処理終了通知用変数 ***#
    global finish

    # 脈波の取得開始時刻の初期化
    generated_pulse_get_start_time = None

    # with open(SAVEFILE_RAW, 'a', newline='') as raw_file:
    #     raw_writer = csv.writer(raw_file, delimiter=',')
    #     raw_writer.writerow(["time", "pulse"])
    with open(SAVEFILE_GENERATED, 'a', newline='') as generated_file:
        generated_writer = csv.writer(generated_file, delimiter=',')
        generated_writer.writerow(["time", "pulse"])

        # 終了フラグが立つまで脈波を取得し続ける
        while not finish:
            try:
                # 脈波値の受信
                read_data = ser.readline().rstrip().decode(encoding='UTF-8')
                # data[0]: micros, data[1]: raw_pulse, data[2]: generated_pulse
                data = read_data.split(",")
                # print(data)

                # 正常値が受信できていることを確認
                if len(data) == 2 and data[0].isdecimal() and data[1].isdecimal():
                    timestamp = float(data[0])/1000

                    #--- データの保存 ---#
                    # raw_writer.writerow([timestamp, int(data[1])])
                    generated_writer.writerow([timestamp, int(data[1])])

                    # センサ値取得時間用キューの更新（単位はミリ秒で保存）
                    # pulse_get_timestamps.append(timestamp)
                    # 生脈波用キューの更新
                    # raw_pulse_values.append(int(data[1]))

                    #--- データセットの作成 ---#
                    # サンプルがSAMPLE_SIZE個貯まるまで待機
                    if train_raw_pulse is None:
                        pulse_get_timestamps, raw_pulse_values = make_train_pulse()

                        # 全サンプルでの点灯時間の取得（最終サンプルのタイムスタンプ - 開始サンプルのタイムスタンプ）
                        display_lighting_time = pulse_get_timestamps[-1] - \
                            pulse_get_timestamps[0]
                        # 学習に使用するデータの取得
                        train_raw_pulse = raw_pulse_values
                        # print('生脈波取得完了')

                    # ディスプレイ点灯開始時に時刻を保存
                    if (send_to_display_data is not None) and (generated_pulse_get_start_time is None):
                        # 脈波の取得開始時刻（データ取得中状態）
                        generated_pulse_get_start_time = timestamp
                        # 取得開始時刻の書き込み
                        generated_writer.writerow([timestamp, 'start'])

                    # データ取得中かつ，擬似脈波受付可能状態の場合
                    if (generated_pulse_get_start_time is not None) and (generated_pulse is None):

                        # 点灯時間（学習データと同じ時間）だけ取得
                        # 現在時刻が(取得開始時刻 + 点灯時間)より大きいかつ，サンプル数が学習データと同じだけ集まったら取得終了
                        if (timestamp > (generated_pulse_get_start_time + display_lighting_time)) and (len(generated_pulse_values) == SAMPLE_SIZE):
                            # ディスプレイ点灯時間の初期化
                            display_lighting_time = None
                            # 脈波の取得開始時刻の初期化
                            generated_pulse_get_start_time = None
                            # 学習用に擬似脈波をコピー
                            generated_pulse = generated_pulse_values

                            # 取得完了時刻の書き込み
                            generated_writer.writerow(
                                [timestamp, 'finish'])
                            # print('擬似脈波取得完了')

                        # 取得時間内
                        else:
                            # 擬似脈波用キューの更新
                            generated_pulse_values.append(int(data[1]))

            except KeyboardInterrupt:
                break


def make_train_pulse():
    """学習用データの作成

    Returns:
        pulse_value (int): 脈波値
    """

    #*** 学習ファイルデータ用変数 ***#
    global train_data

    index = random.randint(1, len(train_data)-SAMPLE_SIZE)

    timestamps = train_data[index:index+SAMPLE_SIZE, 0]
    pulse_data = train_data[index:index+SAMPLE_SIZE, 1]

    return timestamps, pulse_data


def draw_display():
    """ディスプレイの描画

    色データを送信し，ディスプレイに描画．

    Args:
        color (int): 色データ

    Returns:
        pulse_value (int): 脈波値
    """

    #*** 学習生脈波用変数 ***#
    global train_raw_pulse
    #*** データ送信用変数 ***#
    global send_to_display_data
    #*** ディスプレイ点灯時間用変数 ***#
    global display_lighting_time

    train_raw_pulse = None

    while display_lighting_time is None:
        # 1μsの遅延**これを入れないと回りすぎてセンサデータ取得の動作が遅くなる**
        sleep(0.000001)
        continue

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

    #*** ディスプレイ点灯時間用変数 ***#
    global display_lighting_time
    #*** 学習擬似脈波用変数 ***#
    global generated_pulse

    #*** 処理終了通知用変数 ***#
    global finish

    def get_pesudo_pulse(colors):
        """擬似脈波の取得

        Args:
            colors (int): 色データ
        """

        #*** データ送信用変数 ***#
        global send_to_display_data
        #*** 学習擬似脈波用変数 ***#
        global generated_pulse

        # 学習擬似脈波の初期化
        generated_pulse = None

        # ディスプレイ送信用データの作成（Tensorから1次元の整数，文字列のNumpyへ）
        send_to_display_data = np.array(
            colors.detach().cpu().numpy().reshape(-1), dtype=int).astype('str')

        # 描画開始
        # print('描画開始')
        draw_display()
        # print('描画終了')

        # 擬似脈波の取得が完了するまで待機
        while generated_pulse is None:
            # 1μsの遅延**これを入れないと回りすぎてセンサデータ取得の動作が遅くなる**
            sleep(0.000001)
            continue

    '''モデルの構築'''
    model = Pix2Pix(device=device).to(device)

    '''モデルの訓練'''
    # criterion = SoftDTW(gamma=1.0, normalize=True)
    criterion = nn.BCEWithLogitsLoss()
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
        #*** 学習擬似脈波用変数 ***#
        global generated_pulse

        model.D.train()
        model.G.train()

        # ---------------------
        #  生成器の学習
        # ---------------------
        with open(SAVEFILE_GENERATED2, 'a', newline='') as generated_file:
            generated_writer = csv.writer(generated_file, delimiter=',')

            # 生波形から同一の脈波データを生成
            generated_pulse = model.G(raw_pulse)
            if (epoch+1) % 1000 == 0:
                generated_writer.writerow(
                    raw_pulse.to('cpu').detach().numpy().copy().squeeze())
                generated_writer.writerow(
                    generated_pulse.to('cpu').detach().numpy().copy().squeeze())

        # 識別器の学習で使用するためコピー
        generated_pulse_copy = generated_pulse.detach()
        # 擬似脈波に対する識別
        preds = model.D(generated_pulse)
        label = torch.ones(1, 1, SAMPLE_SIZE-54).float().to(
            device)  # 偽物画像のラベルを「本物画像(1)」とする
        loss_G = compute_loss(preds, label)

        #-- 学習 --#
        optimizer_D.zero_grad()
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  識別器の学習
        # ---------------------
        #-- 本物データ --#
        # 生波形に対する識別
        preds = model.D(raw_pulse)
        label = torch.ones(1, 1, SAMPLE_SIZE-54).float().to(device)
        loss_D_real = compute_loss(preds, label)

        #-- 偽物（擬似）データ --#
        # 擬似脈波に対する識別
        preds = model.D(generated_pulse_copy)
        label = torch.zeros(1, 1, SAMPLE_SIZE-54).float().to(device)
        loss_D_fake = compute_loss(preds, label)

        #-- 学習 --#
        loss_D = loss_D_real + loss_D_fake
        optimizer_D.zero_grad()
        optimizer_G.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        return loss_D, loss_G

    '''学習サイクル'''
    for epoch in range(EPOCH_NUM):
        # 生脈波の取得が完了するまで待機
        while train_raw_pulse is None:
            # 1μsの遅延**これを入れないと回りすぎてセンサデータ取得の動作が遅くなる**
            sleep(0.000001)
            continue

        # LSTM入力形式に変換
        tensor_raw = torch.tensor(
            train_raw_pulse, dtype=torch.float, device=device).view(1, 1, -1)
        # 学習
        loss_D, loss_G = train_step(tensor_raw)

        # データの保存
        write_data = [epoch+1, loss_D.item(), loss_G.item()]
        print('Epoch: {}, D Loss: {:.3f}, G Loss: {:.3f}'.format(
            write_data[0], write_data[1], write_data[2]))

        # 毎度クローズしないと，処理中断時に保存されない
        with open(LOSS_DATA, 'a', newline='') as loss_file:
            loss_writer = csv.writer(loss_file, delimiter=',')
            # ヘッダーの書き込み
            if epoch == 0:
                loss_writer.writerow(['Epoch', 'D Loss', 'G Loss'])
            # データの書き込み
            loss_writer.writerow(write_data)

    # 処理終了
    finish = True


if __name__ == '__main__':
    print("\n初期化中...")

    # センサ値取得時間用キュー
    pulse_get_timestamps = deque(maxlen=SAMPLE_SIZE)
    # 生脈波用キュー
    raw_pulse_values = deque(maxlen=SAMPLE_SIZE)

    #*** グローバル：学習ファイルデータ用変数 ***#
    train_data = []
    with open(TRAIN_DATA) as f:
        reader = csv.reader(f)

        # ヘッダーのスキップ
        next(reader)

        for row in reader:
            # データの追加
            train_data.append([float(row[0]), float(row[1])])
    train_data = np.array(train_data)

    # 擬似脈波用キュー
    generated_pulse_values = deque(maxlen=SAMPLE_SIZE)

    #*** グローバル：処理終了通知用変数（センサデータ取得終了の制御） ***#
    finish = False

    #*** グローバル：学習生脈波用変数 ***#
    train_raw_pulse = None
    #*** グローバル：学習擬似脈波用変数 ***#
    generated_pulse = None
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

    # 学習スレッドの開始
    train_thread = threading.Thread(target=main)
    train_thread.setDaemon(True)
    train_thread.start()

    # 脈波取得の開始
    get_pulse()

    # シリアル通信の終了
    ser.close()

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


SAMPLE_SIZE = 500  # サンプルサイズ
EPOCH_NUM = 100000  # 学習サイクル数
KERNEL_SIZE = 13  # カーネルサイズ（奇数のみ）
LAMBDA = 100.0  # 損失の比率パラメータ

FILE_EPOCH_NUM = 500  # 1ファイルに保存するエポック数

now = datetime.datetime.today()
time = now.strftime('%Y%m%d') + '_' + now.strftime('%H%M%S')
SAVE_DIR = './data/' + time + '/'

COLOR_DATA = SAVE_DIR + 'colors.csv'
LOSS_DATA = SAVE_DIR + 'loss.csv'


def get_pulse():
    """脈波の取得
    脈波センサからデータを取得し，データ保存用キューを更新．
    """

    #*** エポック数 ***#
    global epoch

    #*** 学習色データ ***#
    global train_colors
    #*** 学習脈波データ ***#
    global train_pulse

    #*** 処理終了フラグ ***#
    global exit_flag

    # 色データ用キュー
    colors = deque(maxlen=SAMPLE_SIZE)
    # 擬似脈波用キュー
    pulse = deque(maxlen=SAMPLE_SIZE)

    def make_display_data(radian):
        """ランダム色データの生成

        Args:
            radian (int): ラジアン周期
        Returns:
            int: 色データ
        """

        return int((math.sin(5 * math.radians(radian)) + 1) / 2 * 255)

    # 初期化
    timestamp = 0
    radian = 0

    while not exit_flag:
        try:
            # 色データの描画
            color = make_display_data(radian)
            socket_client.send((str(color) + '\0').encode('UTF-8'))
            socket_client.recv(1)
            colors.append(float(color))

            # 脈波値の受信
            read_data = ser.readline().rstrip().decode(encoding='UTF-8')
            # data[0]: micros, data[1]: raw_pulse, data[2]: pulse
            data = read_data.split(',')

            # 正常値が受信できていることを確認
            if len(data) == 2 and data[0].isdecimal() and data[1].isdecimal():
                # 異常値の除外（次の値と繋がって，異常な桁数の場合あり）
                if timestamp != 0 and len(str(int(float(data[0]) / 1000000))) > len(str(int(timestamp))) + 2:
                    continue
                else:
                    pulse.append(int(data[1]))

                    # 取得可能データの作成
                    train_colors = np.array(colors)
                    train_pulse = np.array(pulse)

            # sinの更新
            if radian == 359:
                radian = 0
            else:
                radian += 1

        except KeyboardInterrupt:
            break


def train():
    #*** エポック数 ***#
    global epoch

    #*** 学習色データ ***#
    global train_colors
    #*** 学習脈波データ ***#
    global train_pulse

    #*** 処理終了フラグ ***#
    global exit_flag

    '''モデルの構築'''
    model = Pix2Pix(kernel_size=KERNEL_SIZE, device=device)
    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_Values = nn.L1Loss()
    optimizer_D = optimizers.Adam(model.D.parameters(), lr=0.0002)
    optimizer_G = optimizers.Adam(model.G.parameters(), lr=0.0002)
    model.D.train()
    model.G.train()

    # ones = torch.ones(1, 1, SAMPLE_SIZE).float().to(device)
    # zeros = torch.zeros(1, 1, SAMPLE_SIZE).float().to(device)
    ones = torch.ones(1, 1, SAMPLE_SIZE).to(device)
    zeros = torch.zeros(1, 1, SAMPLE_SIZE).to(device)

    # 学習脈波の取得が完了するまで待機
    while len(train_pulse) != SAMPLE_SIZE:
        sleep(0.000001)

    '''学習サイクル'''
    for epoch in range(1, EPOCH_NUM+1):
        print('\n----- Epoch: ' + str(epoch) + ' -----')

        # 学習データの取得（色データ，脈波データ）
        real_colors = torch.tensor(
            train_colors, dtype=torch.float, device=device).view(1, 1, -1)
        input_pulse = torch.tensor(
            train_pulse, dtype=torch.float, device=device).view(1, 1, -1)

        # ---------------------
        #  生成器の学習
        # ---------------------
        # 色データの生成（G(x)）
        fake_colors = model.G(input_pulse)
        # 識別器の学習で使用するためコピー
        fake_colors_copy = fake_colors.detach()

        out = model.D(torch.cat([fake_colors, input_pulse], dim=1))
        loss_G_GAN = criterion_GAN(out, ones)
        loss_G_Values = criterion_Values(fake_colors, real_colors)

        loss_G = loss_G_GAN + LAMBDA * loss_G_Values
        optimizer_D.zero_grad()
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  識別器の学習
        # ---------------------
        # 本物色データに対する識別
        real_out = model.D(
            torch.cat([real_colors, input_pulse], dim=1))
        loss_D_real = criterion_GAN(real_out, ones)
        # 生成色データに対する識別
        fake_out = model.D(
            torch.cat([fake_colors_copy, input_pulse], dim=1))
        loss_D_fake = criterion_GAN(fake_out, zeros)

        loss_D = loss_D_real + loss_D_fake
        optimizer_D.zero_grad()
        optimizer_G.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # ---------------------
        #  データの保存
        # ---------------------
        write_data = [epoch, loss_D.item(), loss_G.item()]
        print('D Loss: {:.3f}, G Loss: {:.3f}'.format(
            write_data[1], write_data[2]))

        # 毎度クローズしないと，処理中断時に保存されない
        with open(LOSS_DATA, 'a', newline='') as loss_file:
            loss_writer = csv.writer(loss_file, delimiter=',')
            # ヘッダーの書き込み
            if epoch == 1:
                loss_writer.writerow(['Epoch', 'D Loss', 'G Loss'])
            # データの書き込み
            loss_writer.writerow(write_data)
        with open(COLOR_DATA, 'a', newline='') as color_file:
            color_writer = csv.writer(color_file, delimiter=',')
            # ヘッダーの書き込み
            if epoch == 1:
                color_writer.writerow(['Epoch', 'Real', 'Fake'])
            # データの書き込み
            numpy_real_colors = real_colors.detach().cpu().numpy().reshape(-1).astype(int)
            numpy_fake_colors = fake_colors.detach().cpu().numpy().reshape(-1).astype(int)
            for real, fake in zip(numpy_real_colors, numpy_fake_colors):
                color_writer.writerow([epoch, real, fake])

        # ---------------------
        #  モデルの保存
        # ---------------------
        if epoch % FILE_EPOCH_NUM == 0:
            torch.save(model.G.state_dict(),
                       SAVE_DIR + 'model_G_' + str(epoch) + '.pth')
            torch.save(model.D.state_dict(),
                       SAVE_DIR + 'model_D_' + str(epoch) + '.pth')

    # 処理終了
    exit_flag = True


if __name__ == '__main__':
    print('\n初期化中...')

    # ファイル保存ディレクトリの作成
    os.mkdir(SAVE_DIR)

    #*** グローバル：エポック数 ***#
    epoch = 0

    #*** グローバル：学習色データ ***#
    train_colors = np.zeros(0)
    #*** グローバル：学習脈波データ ***#
    train_pulse = np.zeros(0)

    #*** グローバル：処理終了フラグ（センサデータ取得終了の制御） ***#
    exit_flag = False

    # PyTorchの初期化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(1)

    # シリアル通信（Arduino）の初期化
    ser = serial.Serial(USB_PORT, 115200)
    ser.reset_input_buffer()
    sleep(3)  # ポート準備に3秒待機**これがないとシリアル通信がうまく動かない**

    # ソケット通信（Processing）の初期化
    socket_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socket_client.connect((SOCKET_ADDRESS, SOCKET_PORT))

    # 学習スレッドの開始
    train_thread = threading.Thread(target=train)
    train_thread.setDaemon(True)
    train_thread.start()

    # 脈波取得の開始
    get_pulse()

    # シリアル通信の終了
    ser.close()

    print('\n\n----- 学習終了 -----\n\n')
    print('ファイル圧縮中．．．\n\n')

    # ファイルの圧縮
    pulse_module.archive_csv(
        SAVE_DIR + 'colors.csv', step=FILE_EPOCH_NUM, delete_source=True)

    print('結果を描画します．．．')

    # 取得結果の描画
    pulse_module.plot_colors_csv(
        SAVE_DIR, max_epoch=EPOCH_NUM, step=FILE_EPOCH_NUM, savefig=False)
    pulse_module.plot_loss_csv(SAVE_DIR)

    print('\n\n********** 終了しました **********\n\n')

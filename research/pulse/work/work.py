import numpy as np
import torch
import torch.nn as nn
import torch.optim as optimizers
import serial
from time import sleep
import threading
from collections import deque
import socket
from model import Test
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


SAMPLE_SIZE = 100  # サンプルサイズ
EPOCH_NUM = 10000  # 学習サイクル数
KERNEL_SIZE = 13  # カーネルサイズ（奇数のみ）
LAMBDA = 0.0  # 損失の比率パラメータ

INFO_EPOCH = 1000  # 情報を表示するエポック数
SAVE_DATA_STEP = 1000  # ファイルにデータを保存するエポック数

now = datetime.datetime.today()
time = now.strftime('%Y%m%d') + '_' + now.strftime('%H%M%S')
SAVE_DIR = './data/' + time + '/'

COLOR_DATA = SAVE_DIR + 'colors.csv'
LOSS_DATA = SAVE_DIR + 'loss.csv'

TRAIN_DATAS = ['20210228_121559_raw', '20210228_122129_raw',
               '20210228_122727_raw', '20210228_123306_raw',
               '20210228_123855_raw', '20210228_124511_raw']
# TRAIN_DATAS = ['/115200/20210207_121945_raw', '/115200/20210207_122512_raw',
#                '/115200/20210207_123029_raw', '/115200/20210207_123615_raw',
#                '/115200/20210207_154330_raw']


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

    def make_display_data():
        """ランダム色データの生成

        Returns:
            list: 脈波から生成した色データ配列
        """

        #*** 学習ファイルデータ用変数 ***#
        global train_data

        pulse_data = np.array(
            train_data[random.randrange(0, len(train_data) - 1)])
        display_data = np.array(
            pulse_data / max(pulse_data) * 10 + 122, dtype=int)

        return list(display_data)

    display_data = []
    while not exit_flag:
        try:
            # 色データの作成
            if len(display_data) < SAMPLE_SIZE:
                display_data = make_display_data()
                colors.clear()
                pulse.clear()

            # 色データの描画
            color = display_data.pop(0)
            socket_client.send((str(color) + '\0').encode('UTF-8'))
            socket_client.recv(1)
            colors.append(float(color))

            # 脈波値の受信
            data = ser.readline().rstrip().decode(encoding='UTF-8')

            if data.isdecimal() and len(data) <= 3:
                pulse.append(int(data))

                # 取得可能データの作成
                if len(colors) == SAMPLE_SIZE and len(pulse) == SAMPLE_SIZE:
                    train_colors = np.array(colors)
                    train_pulse = np.array(pulse)
            else:
                # 異常値の場合
                continue

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
    model = Test(kernel_size=KERNEL_SIZE, device=device)
    criterion_GAN = nn.BCEWithLogitsLoss()
    # criterion_GAN = nn.MSELoss()
    criterion_Values = nn.L1Loss()
    optimizer_D = optimizers.Adam(model.D.parameters(), lr=0.0002)
    optimizer_G = optimizers.Adam(model.G.parameters(), lr=0.0002)
    model.D.train()
    model.G.train()

    ones = torch.ones(1, 1, SAMPLE_SIZE).to(device)
    zeros = torch.zeros(1, 1, SAMPLE_SIZE).to(device)

    # 学習脈波の取得が完了するまで待機
    while len(train_pulse) != SAMPLE_SIZE:
        sleep(0.000001)

    print('\n*** 学習開始 ***\n')

    '''学習サイクル'''
    for epoch in range(1, EPOCH_NUM+1):
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
        #  データの表示
        # ---------------------
        if epoch % INFO_EPOCH == 0:
            print('\n----- Epoch: ' + str(epoch) + ' -----')
            print('D Loss: {:.3f}, G Loss: {:.3f}\n'.format(
                loss_D.item(), loss_G.item()))
            print('D Real: ')
            print(real_out)
            print('D Fake: ')
            print(fake_out)

        if epoch % SAVE_DATA_STEP == 0:
            # ---------------------
            #  データの保存
            # ---------------------
            write_data = [epoch, loss_D.item(), loss_G.item()]

            # 毎度クローズしないと，処理中断時に保存されない
            with open(LOSS_DATA, 'a', newline='') as loss_file:
                # データの書き込み
                loss_writer = csv.writer(loss_file, delimiter=',')
                loss_writer.writerow(write_data)
            with open(COLOR_DATA, 'a', newline='') as color_file:
                # データの書き込み
                color_writer = csv.writer(color_file, delimiter=',')
                numpy_real_colors = real_colors.detach().cpu().numpy().reshape(-1).astype(int)
                numpy_fake_colors = fake_colors.detach().cpu().numpy().reshape(-1).astype(int)
                numpy_input_pulse = input_pulse.detach().cpu().numpy().reshape(-1).astype(int)
                for real, fake, pulse in zip(numpy_real_colors, numpy_fake_colors, numpy_input_pulse):
                    color_writer.writerow([epoch, real, fake, pulse])

            # ---------------------
            #  モデルの保存
            # ---------------------
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
    # ファイルの用意（ヘッダーの書き込み）
    with open(LOSS_DATA, 'a', newline='') as loss_file:
        loss_writer = csv.writer(loss_file, delimiter=',')
        loss_writer.writerow(['Epoch', 'D Loss', 'G Loss'])
    with open(COLOR_DATA, 'a', newline='') as color_file:
        color_writer = csv.writer(color_file, delimiter=',')
        color_writer.writerow(['Epoch', 'Real', 'Fake', 'Pulse'])

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
    ser = serial.Serial(USB_PORT, 14400)
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
    print('結果を描画します．．．')

    # 取得結果の描画
    pulse_module.plot_data_csv(
        SAVE_DIR, max_epoch=EPOCH_NUM, step=SAVE_DATA_STEP, savefig=False)
    pulse_module.plot_loss_csv(SAVE_DIR)

    print('\n\n********** 終了しました **********\n\n')

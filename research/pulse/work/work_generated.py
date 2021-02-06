import numpy as np
import torch
import torch.nn as nn
import torch.optim as optimizers
import matplotlib.pyplot as plt
import serial
from time import sleep
import threading
from collections import deque
import socket
from model import Pix2Pix
import datetime
import csv
import random
import os
os.chdir(os.path.dirname(__file__))


USB_PORT = 'COM3'

SOCKET_ADDRESS = '192.168.11.2'  # Processingサーバのアドレス
SOCKET_PORT = 10000  # Processingサーバのポート


SAMPLE_SIZE = 512  # サンプルサイズ
EPOCH_NUM = 10000  # 学習サイクル数
KERNEL_SIZE = 13  # カーネルサイズ（奇数のみ）

now = datetime.datetime.today()
time = now.strftime("%Y%m%d") + "_" + now.strftime("%H%M%S")
SAVEFILE_RAW = './data/' + time + "_raw.csv"
SAVEFILE_GENERATED = './data/' + time + "_generated.csv"

TRAIN_DATAS = ['20201202_154312_raw', '20201201_153431_raw', '20210202_133228_raw',
               '20210202_134240_raw', '20210202_135317_raw', '20210202_140046_raw',
               '20210202_140826_raw', '20210202_142202_raw', '20210202_144751_raw',
               '20210202_150358_raw', '20210202_150940_raw', '20210202_153417_raw',
               '20210202_155231_raw', '20210202_162038_raw', '20210202_162739_raw']
LOSS_DATA = './data/' + time + '_loss.csv'


def make_train_pulse():
    """学習用データの作成
    Returns:
        pulse_value (int): 脈波値
    """

    #*** 学習ファイルデータ用変数 ***#
    global train_data

    data = random.randrange(0, len(train_data)-1)
    index = random.randrange(0, len(train_data[data])-SAMPLE_SIZE)

    timestamps = train_data[data][index:index+SAMPLE_SIZE, 0]
    pulse_data = train_data[data][index:index+SAMPLE_SIZE, 1]

    return timestamps, pulse_data


def receive_pulse():
    """脈波の取得
    脈波センサからデータを取得し，データ保存用キューを更新．
    """

    #*** エポック数 ***#
    global epoch

    #*** 学習データ取得フラグ ***#
    global train_get_flag
    #*** 学習脈波用変数 ***#
    global numpy_raw_pulse

    #*** 生成脈波取得開始フラグ ***#
    global generated_get_flag
    #*** 生成脈波取得終了時刻 ***#
    global generated_get_finish

    #*** 処理終了フラグ ***#
    global exit_flag

    # with open(SAVEFILE_RAW, 'a', newline='') as raw_file:
    # raw_writer = csv.writer(raw_file, delimiter=',')
    # raw_writer.writerow(["time", "pulse"])
    with open(SAVEFILE_GENERATED, 'a', newline='') as generated_file:
        generated_writer = csv.writer(generated_file, delimiter=',')
        generated_writer.writerow(['Epoch', 'time', 'pulse'])

        # 終了フラグが立つまで脈波を取得し続ける
        while not exit_flag:
            try:
                # 脈波値の受信
                read_data = ser.readline().rstrip().decode(encoding='UTF-8')
                # data[0]: micros, data[1]: raw_pulse, data[2]: generated_pulse
                data = read_data.split(",")

                # 正常値が受信できていることを確認
                if len(data) == 2 and data[0].isdecimal() and data[1].isdecimal():
                    timestamp = float(data[0]) / 1000
                    # 生脈波の取得
                    timestamps, raw_pulse = make_train_pulse()

                    # 学習データの取得
                    if train_get_flag:
                        if len(raw_pulse) < SAMPLE_SIZE:
                            continue
                        else:
                            numpy_raw_pulse = raw_pulse
                            train_get_flag = False

                    # 生成脈波の取得開始
                    if generated_get_flag:
                        if len(generated_pulse) < SAMPLE_SIZE:
                            # 擬似脈波の追加
                            generated_pulse.append(int(data[1]))
                            #--- データの保存 ---#
                            generated_writer.writerow(
                                [epoch+1, timestamp, int(data[1])])
                        else:
                            # 取得終了時刻
                            generated_get_finish = datetime.datetime.now()
                            generated_get_flag = False

            except KeyboardInterrupt:
                break


def get_generated_pulse(colors):
    """擬似脈波の取得
    Args:
        colors (:obj:`Tensor`): 色データ
    Returns:
        :obj:`Numpy`: 生成脈波
    """

    #*** 生成脈波取得開始フラグ ***#
    global generated_get_flag
    #*** 生成脈波取得終了時刻 ***#
    global generated_get_finish

    # ディスプレイ送信用データの作成（Tensorから1次元の整数，文字列のリストへ）
    display_data = np.array(
        colors.detach().cpu().numpy().reshape(-1), dtype=int).tolist()
    display_data = [str(data) for data in display_data]

    # ---------------------
    #  生成脈波の取得
    # ---------------------
    generated_pulse.clear()
    # 取得開始時刻
    generated_get_start = datetime.datetime.now()
    generated_get_flag = True

    # 描画開始時刻
    draw_start = datetime.datetime.now()

    # 描画（1サンプルずつ送信）
    for data in display_data:
        # 終端文字の追加
        data += '\0'

        # 色データの送信
        socket_client.send(data.encode('UTF-8'))

        # 描画完了通知の待機
        socket_client.recv(1)

    # 描画終了時刻
    draw_finish = datetime.datetime.now()

    # 生成脈波の取得が完了するまで待機
    while generated_get_flag is not False:
        sleep(0.000001)

    print('生成脈波取得時間: ' + str((generated_get_finish - generated_get_start).seconds) + '秒')
    print('描画時間: ' + str((draw_finish - draw_start).seconds) + '秒')

    return np.array(generated_pulse)


def main():
    #*** エポック数 ***#
    global epoch

    #*** 学習データ取得フラグ ***#
    global train_get_flag
    #*** 学習脈波用変数 ***#
    global numpy_raw_pulse

    #*** 処理終了フラグ ***#
    global exit_flag

    '''モデルの構築'''
    model = Pix2Pix(kernel_size=KERNEL_SIZE, device=device)
    # criterion = SoftDTW(gamma=1.0, normalize=True)
    criterion = nn.BCEWithLogitsLoss()
    optimizer_D = optimizers.Adam(model.D.parameters(), lr=0.0002)
    optimizer_G = optimizers.Adam(model.G.parameters(), lr=0.0002)
    model.D.train()
    model.G.train()

    '''学習サイクル'''
    for epoch in range(EPOCH_NUM):
        print('\n----- Epoch: {} -----'.format(epoch+1))

        # 学習データの取得
        numpy_raw_pulse = None
        train_get_flag = True
        while numpy_raw_pulse is None:
            sleep(0.000001)

        # ---------------------
        #  生成器の学習
        # ---------------------
        tensor_raw = torch.tensor(
            numpy_raw_pulse, dtype=torch.float, device=device).view(1, 1, -1)

        # 生波形から同一の脈波データを生成
        colors = model.G(tensor_raw)
        numpy_generated_pulse = get_generated_pulse(colors)

        tensor_generated = torch.tensor(
            numpy_generated_pulse, dtype=torch.float, device=device).view(1, 1, -1)

        # 識別器の学習で使用するためコピー
        tensor_generated_copy = tensor_generated.detach()

        # 擬似脈波に対する識別
        preds = model.D(tensor_generated)
        # 偽物画像のラベルを「本物画像(1)」とする
        label = torch.ones(1, 1, SAMPLE_SIZE).float().to(device)
        loss_G = criterion(preds, label)

        optimizer_D.zero_grad()
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  識別器の学習
        # ---------------------
        # 生波形に対する識別
        preds = model.D(tensor_raw)
        label = torch.ones(1, 1, SAMPLE_SIZE).float().to(device)
        loss_D_real = criterion(preds, label)

        # 生成脈波に対する識別
        preds = model.D(tensor_generated_copy)
        label = torch.zeros(1, 1, SAMPLE_SIZE).float().to(device)
        loss_D_fake = criterion(preds, label)

        loss_D = loss_D_real + loss_D_fake
        optimizer_D.zero_grad()
        optimizer_G.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # ---------------------
        #  データの保存
        # ---------------------
        write_data = [epoch+1, loss_D.item(), loss_G.item()]
        print('D Loss: {:.3f}, G Loss: {:.3f}'.format(
            write_data[1], write_data[2]))

        # 毎度クローズしないと，処理中断時に保存されない
        with open(LOSS_DATA, 'a', newline='') as loss_file:
            loss_writer = csv.writer(loss_file, delimiter=',')
            # ヘッダーの書き込み
            if epoch == 0:
                loss_writer.writerow(['Epoch', 'D Loss', 'G Loss'])
            # データの書き込み
            loss_writer.writerow(write_data)

    # 処理終了
    exit_flag = True


if __name__ == '__main__':
    print("\n初期化中...")

    # センサ値取得時間用キュー
    timestamps = deque(maxlen=SAMPLE_SIZE)
    # 生脈波用キュー
    raw_pulse = deque(maxlen=SAMPLE_SIZE)
    # 擬似脈波用キュー
    generated_pulse = deque(maxlen=SAMPLE_SIZE)

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
                read_data.append([float(row[0]), float(row[1])])
        train_data.append(np.array(read_data))

    #*** グローバル：エポック数 ***#
    epoch = 0

    #*** グローバル：学習データ取得フラグ ***#
    train_get_flag = False
    #*** グローバル：学習生脈波用変数 ***#
    numpy_raw_pulse = None

    #*** グローバル：生成脈波取得開始フラグ ***#
    generated_get_flag = False
    #*** グローバル：生成脈波取得終了時刻 ***#
    generated_get_finish = None

    #*** グローバル：処理終了フラグ（センサデータ取得終了の制御） ***#
    exit_flag = False

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
    receive_pulse()

    # シリアル通信の終了
    ser.close()

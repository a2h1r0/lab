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
import os
os.chdir(os.path.dirname(__file__))


USB_PORT = 'COM3'

SOCKET_ADDRESS = '192.168.11.2'  # Processingサーバのアドレス
SOCKET_PORT = 10000  # Processingサーバのポート


SAMPLE_SIZE = 256  # サンプルサイズ
EPOCH_NUM = 50  # 学習サイクル数
KERNEL_SIZE = 13  # カーネルサイズ（奇数のみ）

FILE_EPOCH_NUM = 10  # 1ファイルに保存するエポック数

now = datetime.datetime.today()
time = now.strftime('%Y%m%d') + '_' + now.strftime('%H%M%S')
SAVE_DIR = './data/' + time
SAVEFILE_RAW = SAVE_DIR + '/raw.csv'
SAVEFILE_GENERATED = SAVE_DIR + '/generated.csv'

TRAIN_DATAS = ['20210207_121945_raw', '20210207_122512_raw',
               '20210207_123029_raw', '20210207_123615_raw',
               '20210207_154330_raw']
LOSS_DATA = SAVE_DIR + '/loss.csv'


def make_train_pulse():
    """学習用データの作成

    Returns:
        pulse_value (int): 脈波値
    """

    #*** 学習ファイルデータ用変数 ***#
    global train_data

    data = random.randrange(0, len(train_data)-1)
    index = random.randrange(0, len(train_data[data])-SAMPLE_SIZE)

    pulse_data = train_data[data][index:index+SAMPLE_SIZE]

    return pulse_data


def get_pulse():
    """脈波の取得
    脈波センサからデータを取得し，データ保存用キューを更新．
    """

    #*** エポック数 ***#
    global epoch

    #*** 生脈波取得フラグ ***#
    global raw_get_flag
    #*** 生脈波用変数 ***#
    global numpy_raw_pulse

    #*** 生成脈波取得開始フラグ ***#
    global generated_get_flag
    #*** 生成脈波用変数 ***#
    global numpy_generated_pulse

    #*** 処理終了フラグ ***#
    global exit_flag

    # 生脈波用キュー
    raw_pulse = deque(maxlen=SAMPLE_SIZE)
    # 擬似脈波用キュー
    generated_pulse = deque(maxlen=SAMPLE_SIZE)

    with open(SAVEFILE_RAW, 'a', newline='') as raw_file:
        raw_writer = csv.writer(raw_file, delimiter=',')
        raw_writer.writerow(['Epoch', 'pulse'])
        with open(SAVEFILE_GENERATED, 'a', newline='') as generated_file:
            generated_writer = csv.writer(generated_file, delimiter=',')
            generated_writer.writerow(['Epoch', 'time', 'pulse'])

            # 終了フラグが立つまで脈波を取得し続ける
            while not exit_flag:
                try:
                    # 脈波値の受信
                    read_data = ser.readline().rstrip().decode(encoding='UTF-8')
                    # data[0]: micros, data[1]: raw_pulse, data[2]: generated_pulse
                    data = read_data.split(',')

                    # 正常値が受信できていることを確認
                    if len(data) == 2 and data[0].isdecimal() and data[1].isdecimal():
                        # 異常値の除外（次の値と繋がって，異常な桁数の場合あり）
                        if 'timestamp' in locals() and len(str(int(float(data[0]) / 1000000))) > len(str(int(timestamp))) + 2:
                            continue

                        timestamp = float(data[0]) / 1000000

                        # 生脈波の取得
                        raw_pulse = make_train_pulse()

                        # 学習データの取得
                        if raw_get_flag:
                            if len(raw_pulse) < SAMPLE_SIZE:
                                continue
                            else:
                                #--- データの保存 ---#
                                for val in raw_pulse:
                                    raw_writer.writerow([epoch+1, int(val)])
                                numpy_raw_pulse = raw_pulse
                                # numpy_raw_pulse = np.array(raw_pulse)
                                raw_get_flag = False

                        # 生成脈波の取得開始
                        if generated_get_flag:
                            # 取得開始時刻
                            if len(generated_pulse) == 0:
                                generated_get_start = timestamp

                            if len(generated_pulse) < SAMPLE_SIZE:
                                # 擬似脈波の追加
                                generated_pulse.append(int(data[1]))
                                #--- データの保存 ---#
                                generated_writer.writerow(
                                    [epoch+1, timestamp, int(data[1])])
                            else:
                                # 取得終了時刻
                                generated_get_finish = timestamp
                                print('生成脈波取得時間: {:.2f}秒'.format(
                                    generated_get_finish - generated_get_start))

                                numpy_generated_pulse = np.array(
                                    generated_pulse)
                                generated_pulse.clear()
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
    #*** 生成脈波用変数 ***#
    global numpy_generated_pulse

    # ディスプレイ送信用データの作成（Tensorから1次元の整数，文字列のリストへ）
    display_data = np.array(
        colors.detach().cpu().numpy().reshape(-1), dtype=int).tolist()
    display_data = [str(data) for data in display_data]

    # ---------------------
    #  生成脈波の取得
    # ---------------------
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

    print('描画時間: {:.2f}秒'.format((draw_finish - draw_start).total_seconds()))

    return numpy_generated_pulse


def main():
    #*** エポック数 ***#
    global epoch

    #*** 生脈波取得フラグ ***#
    global raw_get_flag
    #*** 生脈波用変数 ***#
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
        print('\n----- Epoch: ' + str(epoch+1) + ' -----')

        # 学習データの取得
        numpy_raw_pulse = None
        raw_get_flag = True
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
    print('\n初期化中...')

    # ファイル保存ディレクトリの作成
    os.mkdir(SAVE_DIR)

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

    #*** グローバル：生脈波取得フラグ ***#
    raw_get_flag = False
    #*** グローバル：生脈波用変数 ***#
    numpy_raw_pulse = None

    #*** グローバル：生成脈波取得開始フラグ ***#
    generated_get_flag = False
    #*** グローバル：生成脈波用変数 ***#
    numpy_generated_pulse = None

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
    train_thread = threading.Thread(target=main)
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
        SAVE_DIR + '/generated.csv', step=FILE_EPOCH_NUM, delete_source=True)
    pulse_module.archive_csv(
        SAVE_DIR + '/raw.csv', step=FILE_EPOCH_NUM, delete_source=True)

    print('結果を描画します．．．')

    # 取得結果の描画
    pulse_module.plot_pulse_csv(
        SAVE_DIR, max_epoch=EPOCH_NUM, step=FILE_EPOCH_NUM)
    pulse_module.plot_loss_csv(SAVE_DIR)

    print('\n\n********** 終了しました **********\n\n')

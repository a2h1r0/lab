import matplotlib.pyplot as plt
import csv
from time import sleep
import serial
import socket
import glob
import os
os.chdir(os.path.dirname(__file__))


TIME = '20210317_165322'
SHOW_EPOCH = 5000

USB_PORT = 'COM3'

SOCKET_ADDRESS = '192.168.11.2'  # Processingサーバのアドレス
SOCKET_PORT = 10000  # Processingサーバのポート


def get_pulse(colors):
    """色データの描画と脈波の取得

    Args:
        colors (list): 色データ
    Returns:
        list: 取得した脈波
    """

    pulse = []
    for color in colors:
        # 色データの描画
        socket_client.send((str(color) + '\0').encode('UTF-8'))
        socket_client.recv(1)

        # 脈波値の受信
        data = ser.readline().rstrip().decode(encoding='UTF-8')

        if data.isdecimal() and len(data) <= 3:
            pulse.append(int(data))
        else:
            # 異常値の場合
            continue

    return pulse


print('初期化中．．．\n')

# シリアル通信（Arduino）の初期化
ser = serial.Serial(USB_PORT, 14400)
ser.reset_input_buffer()
sleep(3)  # ポート準備に3秒待機**これがないとシリアル通信がうまく動かない**

# ソケット通信（Processing）の初期化
socket_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
socket_client.connect((SOCKET_ADDRESS, SOCKET_PORT))


# データの読み出し
real = []
fake = []
files = glob.glob('./data/' + TIME + '/colors.csv')
for data in files:
    with open(data) as f:
        reader = csv.reader(f)
        next(reader)

        for row in reader:
            if int(row[0]) < SHOW_EPOCH:
                continue
            elif int(row[0]) == SHOW_EPOCH:
                real.append(int(row[1]))
                fake.append(int(row[2]))
            elif int(row[0]) > SHOW_EPOCH:
                break


# 脈波の取得
print('Real取得中．．．\n')
real_pulse = get_pulse(real)
print('初期化中．．．\n')
sleep(5)
print('Fake取得中．．．\n')
fake_pulse = get_pulse(fake)


# 結果の描画
fig, ax1 = plt.subplots(figsize=(16, 9))
ax1.plot(range(len(real)), real, 'blue', label='Real')
ax1.plot(range(len(fake)), fake, 'red', label='Fake')
ax2 = ax1.twinx()
ax2.plot(range(len(real_pulse)), real_pulse, 'green', label='Real Pulse')
ax2.plot(range(len(fake_pulse)), fake_pulse, 'yellow', label='Fake Pulse')
ax1.set_xlabel('Time [s]', fontsize=18)
ax1.set_ylabel('Gray Scale', fontsize=18)
ax2.set_ylabel('Pulse Value', fontsize=18)
plt.title('Epoch: ' + str(SHOW_EPOCH))
plt.tick_params(labelsize=18)
handler1, label1 = ax1.get_legend_handles_labels()
handler2, label2 = ax2.get_legend_handles_labels()
ax1.legend(handler1 + handler2, label1 + label2,
           fontsize=18, loc='upper right')
# plt.savefig('../figure/10000_generated_' + str(SHOW_EPOCH) + 'epoch.png',
#             bbox_inches='tight', pad_inches=0)

plt.show()

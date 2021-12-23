import csv
import datetime
from time import sleep
import time
import serial
import os
os.chdir(os.path.dirname(__file__))


MODEL = 'SMART_R'  # スマートウォッチのモデル
PROCESS_TIME = 130  # 実行時間（アプリ側のデータ取得は120秒間）
# PROCESS_TIMEはArduino側のプログラムでも設定する必要あり


LOG_FILE = '../../data/' + MODEL + '/run.log'  # ログファイル

USB_PORT = 'COM3'  # ArduinoのUSBポート
USB_SPEED = 9600  # Arduinoの速度


def light(heart_rate):
    """
    点灯

    Args:
        heart_rate (int): 再現する心拍数
    """

    # Arduinoに目標心拍数を送信
    heart_rate = str(heart_rate) + '\0'
    ser.write(heart_rate.encode('UTF-8'))

    # 途中まで経過時間を表示
    start = time.time()
    show_time = 0
    while True:
        process = time.time() - start

        # 10秒ごとに残り時間を表示
        if int(process) % 10 == 0 and int(process) != show_time:
            show_time = int(process)
            print('Remaining... ' + str(PROCESS_TIME - show_time) + 's')

        # 10秒前まで表示
        if process > (PROCESS_TIME - 10):
            break

    # 描画終了を待機
    ser.readline()


if __name__ == '__main__':
    # 心拍数の設定
    heart_rate = input('\n\nTarget Heart Rate > ')

    # シリアル通信（Arduino）の初期化
    ser = serial.Serial(USB_PORT, USB_SPEED)
    ser.reset_input_buffer()
    sleep(2)

    # プログラム実行日時と実行時間，心拍数を記録
    with open(LOG_FILE, 'a', newline='') as log_file:
        log_writer = csv.writer(log_file, delimiter=',')
        now = datetime.datetime.today()
        run_date_time = now.strftime(
            '%Y/%m/%d') + ' ' + now.strftime('%H:%M:%S')
        log_writer.writerow([run_date_time, PROCESS_TIME, heart_rate])

    print('\nDrawing...')

    # 点灯開始
    light(heart_rate)

    print('\n----- Finish -----\n')

    # シリアル通信の終了
    ser.close()

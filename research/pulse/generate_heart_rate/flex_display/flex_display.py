import csv
import datetime
import serial
import os
os.chdir(os.path.dirname(__file__))


MODEL = 'TicWatch'  # スマートウォッチのモデル
# PROCESS_TIME（実行時間）はArduino側のプログラムで設定


LOG_FILE = './data/' + MODEL + '/run.log'  # ログファイル

USB_PORT = 'COM3'  # ArduinoのUSBポート
USB_SPEED = 9600  # Arduinoの速度


def light(heart_rate):
    """点灯

        Args:
            heart_rate (int): 再現する心拍数
    """

    # Arduinoに目標心拍数を送信
    heart_rate = str(heart_rate) + '\0'
    ser.write(heart_rate.encode('UTF-8'))

    # 描画終了を待機
    ser.readline()


if __name__ == '__main__':
    # シリアル通信（Arduino）の初期化
    ser = serial.Serial(USB_PORT, USB_SPEED)
    ser.reset_input_buffer()
    sleep(2)

    # 心拍数の設定
    heart_rate = input('\n\n心拍数は？ > ')

    # プログラム実行日時と実行時間，心拍数を記録
    with open(LOG_FILE, 'a', newline='') as log_file:
        log_writer = csv.writer(log_file, delimiter=',')
        now = datetime.datetime.today()
        run_date_time = now.strftime(
            '%Y/%m/%d') + ' ' + now.strftime('%H:%M:%S')
        log_writer.writerow([run_date_time, PROCESS_TIME, heart_rate])

    print('\n描画中．．．')

    # 点灯開始
    light(heart_rate)

    print('\n----- 描画終了 -----\n')

    # シリアル通信の終了
    ser.close()

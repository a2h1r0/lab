import sys
import serial
from time import sleep


USB_PORT = 'COM3'


def main():
    # Ajaxデータの受け取り
    heart_rate = sys.stdin.read().rstrip()

    # シリアル通信（Arduino）の初期化
    ser = serial.Serial(USB_PORT, 9600)
    ser.reset_input_buffer()
    sleep(2)

    # Arduinoに接続
    ser.write(heart_rate.encode('UTF-8'))
    ser.readline().rstrip().decode(encoding='UTF-8')

    # シリアル通信の終了
    ser.close()


if __name__ == '__main__':
    main()

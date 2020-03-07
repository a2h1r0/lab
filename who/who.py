## 編集箇所 ##
time = 2    # **データ取得時間(秒)**
## ここまで ##

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../modules'))
#from make_traindata import make_traindata as make

import os.path
import csv
import pandas as pd
import serial
from time import sleep
import math
import datetime

        

# Arduinoの用意
ser = serial.Serial('COM4', 57600)
ser2 = serial.Serial('COM3', 57600)        
sleep(1)    # ポート準備に1秒待機**これがないとシリアル通信がうまく動かない**
        
## Arduinoと接続，再実行時はここまで省略 ##
while True:    
    
    input("被ったらEnter:")
    print("\n")
    
    # データの取得と書き込み
    while True:
        # シリアル通信
        ser.write(bytes('1\0', 'utf-8'))
        voltage = ser.readline().decode('UTF-8').rstrip().split()
        ser2.write(bytes('1\0', 'utf-8'))
        voltage2 = ser2.readline().decode('UTF-8').rstrip().split()

        
        # voltage,voltage2の末尾に時間が格納
        del voltage[-1] # voltageの時間は破棄
        
        time = (math.ceil(int(voltage2[-1])/10**4))/10**2   # 時間を秒単位へ変換(小数第2位まで，以下切り上げ)
        del voltage2[-1]    # 経過時間を確認した後，voltage2の時間要素を削除
        
        # データ整列
        voltage.extend(voltage2)                    # 1号機と2号機のデータを結合
        voltage = list(map(float, voltage))
        if sum(voltage) > 145:
            print("no")

            continue
        else:
            ser2.write(bytes('ooyama\0', 'utf-8'))
            print("yes")


    # Arduinoの終了
    ser.close()
    ser2.close()
    
    print("取得終了．\n")
    
    # 再実行の確認
    loop = input("再実行しますか？[y] > ")
    if loop != "y":
        break


print("Finish\n")
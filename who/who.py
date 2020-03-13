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
        
def voltage_get():
    # シリアル通信
    ser.write(bytes('1'+'\0', 'utf-8'))
    voltage = ser.readline().decode('UTF-8').rstrip().split()
    ser2.write(bytes('1'+'\0', 'utf-8'))
    voltage2 = ser2.readline().decode('UTF-8').rstrip().split()
    
    # voltage,voltage2の末尾に時間が格納
    del voltage[-1] # voltageの時間は破棄
    del voltage2[-1]    # 経過時間を確認した後，voltage2の時間要素を削除
    
    # データ整列
    voltage.extend(voltage2)                    # 1号機と2号機のデータを結合
    voltage = list(map(float, voltage))
    
    return voltage    



## Arduinoと接続，再実行時はここまで省略 ##

print("\n\n--------- 開始します ---------\n\n")
ser2.write(bytes('0'+'\0', 'utf-8'))
        
# データの取得と書き込み
while True:
    if sum(voltage_get()) > 145:
        print("ヘルメットを装着してください．\n\n")
        ser2.write(bytes('0'+'\0', 'utf-8'))
        while True:
            voltage = voltage_get()
            if sum(voltage) < 145 or voltage[12] < 1:
                break
            
    else:
        # ここで機械学習

        print("あなたは大山さんですね！\n\n")        
        ser2.write(bytes('2'+'ooyama'+'\0', 'utf-8'))
        while True:
            voltage = voltage_get()
            if sum(voltage) > 145 or voltage[12] < 1:
                break
            
    if voltage[12] < 1:
        break


print("Finish\n")
ser2.write(bytes('3'+'\0', 'utf-8'))

# Arduinoの終了
ser.close()
ser2.close()
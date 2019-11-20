import serial
from time import sleep
import os.path
import csv
import math
import datetime

ser = serial.Serial('COM5', 115200)   # Arduino1号機
ser2 = serial.Serial('COM6', 115200)  # Arduino2号機
filename = 'test.csv'   # データ保存先ファイル
time = 3                # データ取得時間(秒単位)
tester = "tester"       # 正解ラベル(被験者名)
number = 1              # 取得回数


sleep(1)    # ポート準備に1秒待機**これがないとシリアル通信がうまく動かない**

exist = 0                       # 分岐処理用
if os.path.isfile(filename):    # データ保存ファイルの存在確認
    exist = 1
    
with open(filename, 'a', newline='') as f:
    writer = csv.writer(f)

    # ラベルの付与
    if exist == 0:      # ファイルが新規作成の場合，付与する
        writer.writerow(["in0","in1","in2","in3","in4","in5","in6","in7","in8","in9",
                         "inあ","inい","inう","inA","inB","inC",
                         "in10","in11","in12","in13","in14","in15","in16","in17","in18","in19",
                         "inア","inイ","inウ","inD","inE","inF",
                         "Time","Number","Date"])
    elif exist == 1:    # ファイルが既存の場合，付与しない
        print("File exist. Didn't give label.")
        
    date = 0    # 初回判別用変数
    
    # データの取得と書き込み
    while True:
        # シリアル通信
        ser.write("!".encode('UTF-8'))
        voltage = ser.readline().decode('UTF-8').rstrip().split()
        ser2.write("?".encode('UTF-8'))
        voltage2 = ser2.readline().decode('UTF-8').rstrip().split()
        
        # voltage[],voltage2[]の末尾に時間が格納
        del voltage[-1] # voltage[]の時間は破棄
        voltage2[-1] = (math.ceil(int(voltage2[-1])/10**4))/10**2   # 時間を秒単位へ変換(小数第2位まで，以下切り上げ)

        
        # 経過時間がデータ取得時間を超えると，ファイルへ書き込みせずに終了
        if (voltage2[-1] > time):
            break
        
        # データ整列
        voltage.extend(voltage2)                    # 1号機と2号機のデータを結合
        if (date == 0):                             # データ取得開始時，初回のみ
            voltage.append(number)                  # 取得回数を追加
            voltage.append(datetime.date.today())   # 日付を追加
            date += 1
        
        writer.writerow(voltage)    # ファイルへ書き込み

ser.close()
ser2.close()
print("Finish")
import os.path
import csv
import pandas as pd
import serial
from time import sleep
import math
import datetime

time = 5    # データ取得時間(秒)

filename = input("被験者 > ") + ".csv"     # データ保存先
loop = "n"                      # 再実行確認用，デフォルトでは再実行しない
exist = 0                       # ラベル付与の分岐処理用
if os.path.isfile(filename):    # データ保存先の存在確認
    exist = 1
    
with open(filename, 'a', newline='') as f:
    writer = csv.writer(f)

    # ラベルの付与
    if exist == 0:  # ファイルが新規作成の場合，付与する
        writer.writerow(["in0","in1","in2","in3","in4","in5","in6","in7","in8","in9",
                         "inあ","inい","inう","inA","inB","inC",
                         "in10","in11","in12","in13","in14","in15","in16","in17","in18","in19",
                         "inア","inイ","inウ","inD","inE","inF",
                         "Time","Number","Date"])
        number = 1  # 初回取得である
    elif exist == 1:    # ファイルが既存の場合，付与しない
        df = pd.read_csv(filename, usecols=['Number'], encoding='Shift-JIS')
        number = int(df.max()) + 1  # 取得回数の最大値+1回目の取得となる
        
    # Arduinoと接続，再実行時はここまで省略
    while True:
        if (loop == "y"):
            number += 1     # 再実行時には更に+1回目の取得となる
        print(number, "回目のデータ取得です．\n")
        
        ser = serial.Serial('COM5', 57600)
        ser2 = serial.Serial('COM6', 57600)        
        start = 0   # 開始判別用変数
        
        sleep(1)    # ポート準備に1秒待機**これがないとシリアル通信がうまく動かない**
        
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
            if (start == 0):                            # データ取得開始時，初回のみ
                voltage.append(number)                  # 取得回数を追加
                voltage.append(datetime.date.today())   # 日付を追加
                start += 1
            
            # ファイル書き込み
            writer.writerow(voltage)
    
        ser.close()
        ser2.close()
        
        # 再実行の確認
        loop = input("再実行しますか？[y] > ")
        if (loop != "y"):
            break
        input("被り直したらEnter:")
        print("\n")
        
print("Finish\n")
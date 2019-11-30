import os.path
import csv
import pandas as pd
import sys
from operator import add

tester = input("計算したい被験者 > ")
readfile = tester + ".csv"
writefile = tester + "_ave.csv"     # データ保存先
if os.path.isfile(writefile):    # データ保存先の存在確認
    print("平均値ファイルが存在します．\n")
    sys.exit()
    
with open(writefile, 'a', newline='') as f:  # 保存先をオープン
    writer = csv.writer(f)

    ## ラベルの付与と取得回数の確認 ##
    writer.writerow(["in0","in1","in2","in3","in4","in5","in6","in7",
                     "in8","in9","inあ","inい","inう","inA","inB","inC",
                     "in10","in11","in12","in13","in14","in15","in16",
                     "in17","in18","in19","inア","inイ","inウ","inD","inE",
                     "inF","Number"])

        ## データの読み込み ##
    data = [] # データ配列，被験者数分用意
    data = pd.read_csv(readfile, usecols=["in0","in1","in2","in3","in4","in5","in6","in7",
                                            "in8","in9","inあ","inい","inう","inA","inB","inC",
                                            "in10","in11","in12","in13","in14","in15","in16",
                                            "in17","in18","in19","inア","inイ","inウ","inD","inE",
                                            "inF","Number"], encoding='Shift-JIS')
    data.fillna(0, inplace=True) # 区切り番号以外"0"で埋める
    sensors = len(data.iloc[0])-1

    
    ## データの計算 ##
    # 各データ，各取得回ごとに平均値を計算
    # 被験者変更時に変数を初期化
    vector_temp = [0]*sensors    # ベクトルの合計
    num = 0         # データ数(計算回数)，取得回数が一定ではないためカウントが必要
    
    # 平均値の計算
    for row in data.itertuples(name=None):   ## 1行ずつ読み出し
        row = list(row)
        # 区切りごとに平均値を保存，変数を初期化
        if (row[-1] != 0):  # 区切りの検出
            n = int(row[-1])    # 区切りを代入
            del row[-1]
            if (n != 1):    # 最初の区切り"1"ではスキップ
                ## 平均値を計算(要素ごとに除算していく)
                vector_temp = [item/num for item in vector_temp]
                vector_temp.append(n-1)
                writer.writerow(vector_temp)
                vector_temp = [0]*sensors    # 変数の初期化
                num = 0
            
        del row[0]
        vector_temp = list(map(add, vector_temp, row))
        num += 1                            # データ数(計算回数)を増加
        
    # 最終データの平均値を保存
    vector_temp = [item/num for item in vector_temp]
    vector_temp.append(n)
    writer.writerow(vector_temp)
    # 区切り文字なしでデータが終了するため

print("finish")
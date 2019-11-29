## データにより編集 ##
tester = ["fujii", "ooyama", "okamoto", "kajiwara", "matsuda"] # **被験者**
train_size = 2      # **学習に当てる個数**
MIN = 1.0       # **閾値の下限**
MAX = 1.1       # **閾値の上限**
digit = 10
get_num = 18    # データ取得回数の最大値
## ここまで随時変更．閾値の桁数を変更する場合は以下コードも変更． ##



sensors = 32


import pandas as pd
import numpy as np
import itertools
from operator import add
from operator import sub
from statistics import mean
import matplotlib.pyplot as plt

thresholds = np.linspace(MIN, MAX, int((MAX-MIN)*digit+1))  # 閾値の配列をx軸として作成


## データの読み込み ##
data = [[] for i in tester] # データ配列，被験者数分用意
for i, name in enumerate(tester):            # 被験者1人ずつ読み込む
    filename = name + ".csv"
    data[i] = pd.read_csv(filename, usecols=["in0","in1","in2","in3","in4","in5","in6","in7",
                                            "in8","in9","inあ","inい","inう","inA","inB","inC",
                                            "in10","in11","in12","in13","in14","in15","in16",
                                            "in17","in18","in19","inア","inイ","inウ","inD","inE",
                                            "inF","Number"], encoding='Shift-JIS')
    data[i].fillna(0, inplace=True) # 区切り番号以外"0"で埋める


## データの計算 ##
# 各データ，各取得回ごとに平均値を計算
#vector_ave = np.zeros((len(tester), get_num, sensors))   # vector_ave[被験者][取得回数][センサ番号(ベクトル要素)]
vector_ave = [[] for i in tester]
for order in range(len(tester)):    ## 被験者ごとに順番に処理
    # 被験者変更時に変数を初期化
    vector_sum = [0]*sensors    # ベクトルの合計
    n = 0   # 保存先指定用カウンタ(取得回数ごとに0から格納)
    num = 0         # データ数(計算回数)，取得回数が一定ではないためカウントが必要

    # 平均値の計算
    for row in data[order].itertuples(name=None):   ## 1行ずつ読み出し
        row = list(row)
        # 区切りごとに平均値を保存，変数を初期化
        if (row[-1] != 0):  # 区切りの検出
            n = int(row[-1])    # 区切りを代入
            del row[-1]
            if (n != 1):    # 最初の区切り"1"ではスキップ
                ## 平均値を計算(要素ごとに除算していく)
                vector_sum = [item/num for item in vector_sum]
                vector_ave[order].append(vector_sum)                
                vector_sum = [0 for i in range(sensors)]    # 変数の初期化
                num = 0
            
        del row[0]
        vector_sum = list(map(add, vector_sum, row))
        num += 1                            # データ数(計算回数)を増加
        
    # 最終データの平均値を保存
    vector_sum = [item/num for item in vector_sum]
    vector_ave[order].append(vector_sum)    # 区切り文字なしでデータが終了するため
    
    

## データの類似度計算と，判定 ##
# 被験者数分の結果用配列を作成
FRR = [[] for i in range(len(tester))] # 本人拒否率
FAR = [[] for i in range(len(tester))] # 他人受入率
# ループ用に変数の調整
int_MIN = int(MIN*digit)      # 整数化
int_MAX = int(MAX*digit)+1    # 範囲用に+1
    
# 計算と判定
for threshold in thresholds:   ## 閾値の移動
    print("閾値は"+str(threshold)+"です．")
       
    for index, trainer in enumerate(tester):    ## 1人ずつ学習データにする
        print(trainer+"が学習データです．")    
        combinations = list(itertools.combinations(np.arange(len(vector_ave[index])), train_size))    # 組み合わせの取得
        print("組み合わせはk="+str(len(combinations))+"通りです．")
        
        # 全ての組み合わせについて計算していく
        FRR_temp = np.zeros(len(combinations))  # 一時保存用の配列を作成
        FAR_temp = np.zeros(len(combinations))  # 組み合わせごとに結果を保存
           
        for order, combination in enumerate(combinations):  ## 組み合わせの変更，交差検証            
            num_trainer = 0 # 判別回数の初期化
            num_attacker = 0
               
            for attacker in tester:   ## 1人ずつ認証データにする                
                for item, vector in enumerate(vector_ave[index]):   ## 1データずつ判別
                    if (attacker == trainer and item in combination): # 本人のデータと判別かつ現在の組み合わせに含まれる場合
                        continue                                       # 認証と学習データが同一のためスキップ
                        
                    distance = []
                    for train in combination:     ## 1データにつき，学習データそれぞれと比較(学習データ数回比較)
#                        for k in range(sensors):
#                            distance += abs(vector_ave[train][i][k]-vector_ave[train][combinations[order][j]][k])
                        distance.append(sum(np.abs(list(map(sub, vector, vector_ave[index][train])))))
# =============================================================================
#                         if (distance < distance_small):
#                             distance_small = distance
#                             distance = 0
#                             print(distance_small)
# =============================================================================
                           
# =============================================================================
#                         # ベクトルのノルム(大きさ)の差の絶対値を計算
#                         # norm_ave[train][combinations[order][j]]で学習データを指定(組み合わせの中身の番号を順に取得)
#                         distance = abs(norm_ave[train][i]-norm_ave[train][combinations[order][j]])
#                         if (distance < distance_small): # 比較した中で差が最小のもの(最も類似している)を結果(差)とする
#                             distance_small = distance
# =============================================================================

                    if (attacker==trainer):
                        num_trainer += 1
                        if (min(distance)>threshold):
                            FRR_temp[order] += 1
                    elif (attacker!=trainer):
                        num_attacker += 1
                        if (min(distance)<=threshold):
                            FAR_temp[order] += 1

# =============================================================================
#                     # 範囲を整数化しているので，小数に戻して比較
#                     if (min(distance)<=threshold and attacker!=trainer):     # 差が閾値以下なら，受け入れる
#                         FAR_temp[order] += 1 # 他人受入数
#                     elif (min(distance)>threshold and attacker==trainer):    # 閾値より大きいなら，弾く
#                         FRR_temp[order] += 1 # 本人拒否数
#                     num += 1    # 判別回数を増加
#                        
# =============================================================================
            # 判別終了後，組み合わせ変更時に確率計算
            # (学習データのデータ数-学習に当てた個数)の残りの数だけ本人と判別
#            num = num-(len(vector_ave[index])-train_size) # (総判別回数-本人との判別回数)で他人との判別回数
#            FRR_temp[order] = (FRR_temp[order]/(len(vector_ave[index])-train_size))*100 # 本人と判別した内，拒否した割合
            FRR_temp[order] = (FRR_temp[order]/num_trainer)*100 # 本人と判別した内，拒否した割合
            FAR_temp[order] = (FAR_temp[order]/num_attacker)*100   # 他人と判別した内，受け入れた割合
               
        # 学習データに用いる被験者の変更時に結果を保存
        FRR[index].append(mean(FRR_temp))   # 交差検証を行った結果の平均を被験者ごとのリストで保存
        FAR[index].append(mean(FAR_temp))   # 閾値に対する結果を要素として追加していく         
        print("FRR:"+str(FRR[index][-1]))        # 最新は末尾
        print("FAR:"+str(FAR[index][-1]))
        print("\n----------\n")
   
    
        
## 結果の描画 ##
#x = np.linspace(MIN, MAX, int_MAX-int_MIN)  # 閾値の配列をx軸として作成
for train in range(len(tester)):    ## 被験者ごとに描画
    plt.figure(train)   # 複数ウィンドウで表示
    plt.xlabel("Threshold")
    plt.ylabel("Rate")
    plt.title(tester[train])
    plt.plot(thresholds, FRR[train], 'red', label="FRR")
    plt.plot(thresholds, FAR[train], 'blue', label="FAR")
    plt.legend()    # 凡例の表示
plt.show()
   
## ここまで完成 ##
   
x = range(32)
plt.xlabel("sensor")
plt.ylabel("voltage")
plt.title("vector_norm")
for i, name in enumerate(tester):
    for j in range(len(vector_ave[i])):
        plt.figure(i+len(tester))
        plt.title(name)
        plt.plot(x, vector_ave[i][j], 'red')
plt.show()

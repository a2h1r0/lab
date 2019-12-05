## データにより編集 ##
tester = ["fujii", "ooyama", "okamoto", "kajiwara", "matsuda"] # **被験者**
train_size = 2      # **学習に当てる個数**
MIN = 0.00       # **閾値の下限**
MAX = 1.00       # **閾値の上限**
digit = 100
## ここまで随時変更．閾値の桁数を変更する場合は以下コードも変更． ##




import pandas as pd
import numpy as np
import itertools
from operator import add
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
import warnings


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
    sensors = len(data[0].iloc[0])-1


## データの計算 ##
# 各データ，各取得回ごとに平均値を計算
vector_ave = [[] for i in tester]
for order in range(len(tester)):    ## 被験者ごとに順番に処理
    # 被験者変更時に変数を初期化
    vector_temp = [0]*sensors    # ベクトルの合計
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
                vector_temp = [item/num for item in vector_temp]
                vector_ave[order].append(vector_temp)                
                vector_temp = [0]*sensors    # 変数の初期化
                num = 0
            
        del row[0]
        vector_temp = list(map(add, vector_temp, row))
        num += 1                            # データ数(計算回数)を増加
        
    # 最終データの平均値を保存
    vector_temp = [item/num for item in vector_temp]
    vector_ave[order].append(vector_temp)    # 区切り文字なしでデータが終了するため
  

## データの類似度計算と，判定 ##
# 自分と比較
warnings.simplefilter('ignore')

clf = LocalOutlierFactor(n_neighbors=2)
FRR_num = 0
FAR_num = 0
FRR = []
FAR = []
for index_train, train in enumerate(tester):
    train_data = []
    train_data = vector_ave[index_train][0:16]
    num_train = 0
    num_attack = 0
    for index_attack, attack in enumerate(tester): ## 攻撃データ
        for vector in vector_ave[index_attack]:   # 1データずつ取り出し
            train_data.append(vector)        
            pred = clf.fit_predict(train_data)
            if (attack==train):
                num_train += 1
                if (pred[-1] == -1):
                    FRR_num += 1
            elif (attack!=train):
                num_attack += 1
                if (pred[-1] == 1):
                    FAR_num += 1
            del train_data[-1]
                    
    FRR.append((FRR_num/num_train)*100)
    FAR.append((FAR_num/num_attack)*100)



    
"""        
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
"""
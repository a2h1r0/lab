## データにより編集 ##
tester = ["ooyama", "okamoto", "kajiwara", "fujii", "matsuda"] # **被験者**
train_size = 2      # **学習に当てる個数**
MIN = 0       # **閾値の下限**
MAX = 600       # **閾値の上限**
digit = 1
k = 5
## ここまで随時変更．閾値の桁数を変更する場合は以下コードも変更． ##




import pandas as pd
import numpy as np
from operator import add
import matplotlib.pyplot as plt
from sklearn.covariance import MinCovDet

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

mcd = MinCovDet()
FRR = np.zeros((len(tester), len(thresholds)))
FAR = np.zeros((len(tester), len(thresholds)))
for index_train, train in enumerate(tester):    ## 学習データ
    data_size = int(len(vector_ave[index_train])/k) # データサイズの計算
    for order in range(k): #交差検証，テストデータを選択
        FRR_num = np.zeros(len(thresholds))
        FAR_num = np.zeros(len(thresholds))
        train_data = []
        attack_data = []
        for i in range(k):  ##学習データの作成
            if (i != order):
                train_data.extend(vector_ave[index_train][i*data_size:(i+1)*data_size])
        for i in range(k):  ##攻撃(正解)データの作成
            if (i == order):
                attack_data.extend(vector_ave[index_train][i*data_size:(i+1)*data_size])
        for index_attack, attack in enumerate(tester): ## 被験者変更
            if (attack != train):
                attack_data.extend(vector_ave[index_attack])
          
        # MCD
        mcd.fit(train_data)
        anomaly_score_mcd = mcd.mahalanobis(attack_data)
        
        for index, threshold in enumerate(thresholds):
            num_train = 0
            num_attack = 0            
            for item, distance in enumerate(anomaly_score_mcd):
                if (item<data_size and distance>threshold):
                        FRR_num[index] += 1
                elif (item>=data_size and distance<threshold):
                        FAR_num[index] += 1
            
        FRR[index_train] += FRR_num/data_size
        FAR[index_train] += FAR_num/(len(anomaly_score_mcd)-data_size)
        
FRR = FRR/k*100
FAR = FAR/k*100
FRR_total = FRR.mean(axis=0)
FAR_total = FAR.mean(axis=0)
            
            
## 結果の描画 ##
plt.figure(0)   # 複数ウィンドウで表示
plt.title("total")
plt.plot(thresholds, FRR_total, 'red', label="FRR")
plt.plot(thresholds, FAR_total, 'blue', label="FAR")
plt.xlabel("Threshold")
plt.ylabel("Rate")
plt.legend()    # 凡例の表示
#for train in range(1, len(tester)+1):    ## 被験者ごとに描画
#    plt.figure(train)   # 複数ウィンドウで表示
#    plt.title(tester[train-1])
#    plt.plot(thresholds, FRR[train-1], 'red', label="FRR")
#    plt.plot(thresholds, FAR[train-1], 'blue', label="FAR")
#    plt.xlabel("Threshold")
#    plt.ylabel("Rate")
#    plt.legend()    # 凡例の表示
plt.show()
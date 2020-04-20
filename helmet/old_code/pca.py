## データにより編集 ##
tester = ["fujii", "ooyama", "okamoto", "kajiwara", "matsuda"] # **被験者**
MIN = 0.00       # **閾値の下限**
MAX = 3.00       # **閾値の上限**
digit = 100
split_size = 2
## ここまで随時変更．閾値の桁数を変更する場合は以下コードも変更． ##




import pandas as pd
import numpy as np
from operator import add
import matplotlib.pyplot as plt
from sklearn import decomposition

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
#vector_ave = np.zeros((len(tester), get_num, sensors))   # vector_ave[被験者][取得回数][センサ番号(ベクトル要素)]
model = decomposition.PCA(n_components=2)
vector_ave = [[] for i in tester]
compressed = [[] for i in tester]
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
    
    model.fit(vector_ave[order])
    compressed[order] = model.transform(vector_ave[order])


## データの類似度計算と，判定 ##
# 被験者数分の結果用配列を作成
FRR = [[] for i in range(len(tester))] # 本人拒否率
FAR = [[] for i in range(len(tester))] # 他人受入率
# ループ用に変数の調整
int_MIN = int(MIN*digit)      # 整数化
int_MAX = int(MAX*digit)+1    # 範囲用に+1

center = [[[np.zeros(2)] for i in range(split_size)] for i in range(len(tester))]
    
# 計算と判定
for index, trainer in enumerate(tester):    ## 1人ずつ学習データにする
    print(trainer+"が学習データです．")    
        
    # 全ての組み合わせについて計算していく
    FRR_num = np.zeros(len(thresholds))  # 一時保存用の配列を作成
    FAR_num = np.zeros(len(thresholds))  # 組み合わせごとに結果を保存
    FRR_temp = [[] for i in range(split_size)]  # 一時保存用の配列を作成
    FAR_temp = [[] for i in range(split_size)]  # 組み合わせごとに結果を保存

    train_size = int(len(compressed[index])/split_size)
    combination = 0
           
        
    for order in range(split_size):  ## 組み合わせの変更，交差検証            
        num_trainer = 0 # 判別回数の初期化
        num_attacker = 0
            
        # 重心計算
        for item in range(combination, train_size+combination):
            center[index][order] += compressed[index][item]
            combination += 1
        center[index][order] /= train_size
        
        
               
## 自分と比較        
for index, trainer in enumerate(tester):   ## 1人ずつが学習データにする
    for i, gravity in enumerate(center[index]): #交差検証
        for index_atk, attacker in enumerate(tester): ## 攻撃データ
            for item, vector in enumerate(compressed[index_atk]):   # 1データずつ取り出し
                distance = np.linalg.norm(vector-gravity)
                if (attacker==trainer):
                    num_trainer += 1
                elif (attacker!=trainer):
                    num_attacker += 1
  
                for j, threshold in enumerate(thresholds):    # 閾値
                        if (attacker==trainer and distance>threshold):
                            FRR_num[j] += 1
                        if (attacker!=trainer and distance<=threshold):
                            FAR_num[j] += 1
                            
        FRR_temp[i] = FRR_num/num_trainer
        FAR_temp[i] = FAR_num/num_attacker
        
        
    for i in range(len(thresholds)):
        FRR[index].append(((FRR_temp[0][i]+FRR_temp[1][i])/2)*100)
        FAR[index].append(((FAR_temp[0][i]+FAR_temp[1][i])/2)*100)
            
    
        
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
   
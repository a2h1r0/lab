## データにより編集 ##
tester = ["ooyama", "okamoto", "kajiwara", "fujii", "matsuda"] # **被験者**
train_size = 2      # **学習に当てる個数**
MIN = 0.00       # **閾値の下限**
MAX = 1.00       # **閾値の上限**
digit = 100
## ここまで随時変更．閾値の桁数を変更する場合は以下コードも変更． ##




import pandas as pd
import numpy as np
import itertools
from operator import add
from operator import truediv
from statistics import mean
import matplotlib.pyplot as plt
from sklearn import decomposition
import sys

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
model = decomposition.PCA(n_components=14)
vector_ave = [[] for i in tester]
compressed = [[] for i in tester]
vector_sum = []
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
    
    vector_sum.append(vector_ave[order])
    
    model.fit(vector_ave[order])
    compressed[order] = model.transform(vector_ave[order])


## データの類似度計算と，判定 ##
# 被験者数分の結果用配列を作成
FRR = []# 本人拒否率
FAR = [] # 他人受入率
# ループ用に変数の調整
int_MIN = int(MIN*digit)      # 整数化
int_MAX = int(MAX*digit)+1    # 範囲用に+1
    
# 計算と判定
       
centers = [[] for i in tester]

for index, train in enumerate(tester):    ## 1人ずつ学習データにする
    print(train+"の重心を計算します．")    
    combinations = list(itertools.combinations(np.arange(len(vector_ave[index])), train_size))    # 組み合わせの取得
    print("組み合わせはk="+str(len(combinations))+"通りです．")
            
    centers[index] = [[np.zeros(14)] for i in combinations]       
    
    for order, combination in enumerate(combinations):  ## 組み合わせの変更，交差検証                        
        # 重心計算
        for item in combination:
            centers[index][order] += compressed[index][item]
        centers[index][order] /= train_size


              
## 自分と比較        
for item, threshold in enumerate(thresholds):    # 閾値
    FRR_num = 0
    FAR_num = 0
    num_train = 0
    num_attack = 0
    distance_me = []
    distance_other = []
    for index_attack, attack in enumerate(tester): ## 攻撃データ
        for vector in compressed[index_attack]:   # 1データずつ取り出し
            distance = np.linalg.norm(vector-centers[0][0])
            if (attack==train):
                num_train += 1
                distance_me.append(distance)
                if (distance>threshold):
                    FRR_num += 1
            elif (attack!=train):
                num_attack += 1
                distance_other.append(distance)
                if (distance<=threshold):
                    FAR_num += 1        
        
    FRR.append((FRR_num/num_train)*100)
    FAR.append((FAR_num/num_attack)*100)
    
    
# =============================================================================
#     for i in range(len(thresholds)):
#         FRR[index] = FRR_temp
#         
#         FRR[index].append(((FRR_temp[0][i]+FRR_temp[1][i])/2)*100)
#         FAR[index].append(((FAR_temp[0][i]+FAR_temp[1][i])/2)*100)
#  
# 
#             
#             
#                
#             for attacker in tester:   ## 1人ずつ認証データにする                
#                 for item, vector in enumerate(compressed[index]):   ## 1データずつ判別
#                     if (attacker == trainer and item in combination): # 本人のデータと判別かつ現在の組み合わせに含まれる場合
#                         continue                                       # 認証と学習データが同一のためスキップ
#                         
#                     distance = np.linalg.norm(compressed[index]-center)
#                     if (attacker==trainer):
#                         num_trainer += 1
#                         if (distance>threshold):
#                             FRR_temp[order] += 1
#                     elif (attacker!=trainer):
#                         num_attacker += 1
#                         if (distance<=threshold):
#                             FAR_temp[order] += 1
# 
#             FRR_temp[order] = (FRR_temp[order]/num_trainer)*100 # 本人と判別した内，拒否した割合
#             FAR_temp[order] = (FAR_temp[order]/num_attacker)*100   # 他人と判別した内，受け入れた割合
#                
#         # 学習データに用いる被験者の変更時に結果を保存
#         FRR[index].append(mean(FRR_temp))   # 交差検証を行った結果の平均を被験者ごとのリストで保存
#         FAR[index].append(mean(FAR_temp))   # 閾値に対する結果を要素として追加していく         
#         print("FRR:"+str(FRR[index][-1]))        # 最新は末尾
#         print("FAR:"+str(FAR[index][-1]))
#         print("\n----------\n")
#    
#     
# =============================================================================
        
## 結果の描画 ##
#x = np.linspace(MIN, MAX, int_MAX-int_MIN)  # 閾値の配列をx軸として作成
plt.xlabel("Threshold")
plt.ylabel("Rate")
plt.plot(thresholds, FRR, 'red', label="FRR")
plt.plot(thresholds, FAR, 'blue', label="FAR")
plt.legend()    # 凡例の表示
plt.show()
   
## ここまで完成 ##
   
# =============================================================================
# x = range(32)
# plt.xlabel("sensor")
# plt.ylabel("voltage")
# plt.title("vector_norm")
# for i, name in enumerate(tester):
#     for j in range(len(vector_ave[i])):
#         plt.figure(i+len(tester))
#         plt.title(name)
#         plt.plot(x, vector_ave[i][j], 'red')
# plt.show()
# 
# =============================================================================

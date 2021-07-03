## データにより編集 ##
tester = ["fujii", "ooyama", "okamoto", "kajiwara", "matsuda"] # **被験者**
train_size = 2      # **学習に当てる個数**
MIN = 0.100       # **閾値の下限**
MAX = 1.500       # **閾値の上限**
## ここまで随時変更．閾値の桁数を変更する場合は以下コードも変更． ##



import pandas as pd
import numpy as np
import itertools
from statistics import mean
import matplotlib.pyplot as plt

## データの読み込み ##
data = [[] for i in range(len(tester))] # データ配列，被験者数分用意
for i in range(len(tester)):            # 被験者1人ずつ読み込む
    filename = tester[i] + ".csv"
    data[i] = pd.read_csv(filename, usecols=["in0","in1","in2","in3","in4","in5","in6","in7",
                                            "in8","in9","inあ","inい","inう","inA","inB","inC",
                                            "in10","in11","in12","in13","in14","in15","in16",
                                            "in17","in18","in19","inア","inイ","inウ","inD","inE",
                                            "inF","Number"], encoding='Shift-JIS')
    data[i].fillna(0, inplace=True) # 区切り番号以外"0"で埋める


## データの計算 ##
# 各データ，各取得回ごとに平均値を計算
norm_ave = [[] for i in range(len(tester))] # 平均値配列，被験者数分用意
for order in range(len(tester)):    ## 被験者ごとに順番に処理
    # 被験者変更時に変数を初期化
    num = 0         # データ数(計算回数)，取得回数が一定ではないためカウントが必要
    norm_sum = 0    # ベクトルの合計

    # 平均値の計算
    for row in data[order].itertuples(name=None):   ## 1行ずつ読み出し
        # 区切りごとに平均値を保存，変数を初期化
        if not (row[-1] in [0,1]):  # 区切りではない"0"と最初の区切り"1"ではスキップ
            norm_ave[order].append(norm_sum/num)
            num = 0
            norm_sum = 0
        # ベクトルのノルム(大きさ)を計算
        norm = np.linalg.norm(row[1:34])    # row[0]にはデータ番号が格納されている
        norm_sum += norm                    # ベクトルの合計に加算
        num += 1                            # データ数(計算回数)を増加
    # 最終データの平均値を保存
    norm_ave[order].append(norm_sum/num)    # 区切り文字なしでデータが終了するため


## データの類似度計算と，判定 ##
# 被験者数分の結果用配列を作成
FRR = [[] for i in range(len(tester))] # 本人拒否率
FAR = [[] for i in range(len(tester))] # 他人受入率
# ループ用に変数の調整
int_MIN = int(MIN*1000)      # 整数化
int_MAX = int(MAX*1000)+1    # 範囲用に+1

# 計算と判定
for threshold in range(int_MIN, int_MAX):   ## 閾値の移動
    print("閾値は"+str(threshold/1000)+"です．")
    
    for train in range(len(tester)):    ## 1人ずつ学習データにする
        print(tester[train]+"が学習データです．")    
        combinations = list(itertools.combinations(range(len(norm_ave[train])), train_size))    # 組み合わせの取得
        print("組み合わせはk="+str(len(combinations))+"通りです．")
        
        # 全ての組み合わせについて計算していく
        FRR_temp = np.zeros(len(combinations))  # 一時保存用の配列を作成
        FAR_temp = np.zeros(len(combinations))  # 組み合わせごとに結果を保存
        
        for order in range(len(combinations)):  ## 組み合わせの変更，交差検証            
            num = 0 # 判別回数の初期化
            
            for attack in range(len(tester)):   ## 1人ずつ認証データにする                
                for i in range(len(norm_ave[train])):   ## 1データずつ判別
                    if (attack == train and i in combinations[order]): # 本人のデータと判別かつ現在の組み合わせに含まれる場合
                        continue                                       # 認証と学習データが同一のためスキップ
                        
                    distance_small = float('inf')   # 最小比較用に無限大で初期化
                    for j in range(train_size):     ## 1データにつき，学習データそれぞれと比較(学習データ数回比較)
                        # ベクトルのノルム(大きさ)の差の絶対値を計算
                        # norm_ave[train][combinations[order][j]]で学習データを指定(組み合わせの中身の番号を順に取得)
                        distance = abs(norm_ave[train][i]-norm_ave[train][combinations[order][j]])
                        if (distance < distance_small): # 比較した中で差が最小のもの(最も類似している)を結果(差)とする
                            distance_small = distance
                  
                    # 範囲を整数化しているので，小数に戻して比較
                    if (distance_small<=(threshold/1000) and attack!=train):     # 差が閾値以下なら，受け入れる
                        FAR_temp[order] += 1 # 他人受入数
                    elif (distance_small>(threshold/1000) and attack==train):    # 閾値より大きいなら，弾く
                        FRR_temp[order] += 1 # 本人拒否数
                    num += 1    # 判別回数を増加
                    
            # 判別終了後，組み合わせ変更時に確率計算
            # (学習データのデータ数-学習に当てた個数)の残りの数だけ本人と判別
            num = num-(len(norm_ave[train])-train_size) # (総判別回数-本人との判別回数)で他人との判別回数
            FRR_temp[order] = (FRR_temp[order]/(len(norm_ave[train])-train_size))*100 # 本人と判別した内，拒否した割合
            FAR_temp[order] = (FAR_temp[order]/num)*100   # 他人と判別した内，受け入れた割合
            
        # 学習データに用いる被験者の変更時に結果を保存
        FRR[train].append(mean(FRR_temp))   # 交差検証を行った結果の平均を被験者ごとのリストで保存
        FAR[train].append(mean(FAR_temp))   # 閾値に対する結果を要素として追加していく         
        print("FRR:"+str(FRR[train][-1]))        
        print("FAR:"+str(FAR[train][-1]))
        print("\n----------\n")

     
## 結果の描画 ##
x = np.linspace(MIN, MAX, int_MAX-int_MIN)  # 閾値の配列をx軸として作成
for train in range(len(tester)):    ## 被験者ごとに描画
    plt.figure(train)   # 複数ウィンドウで表示
    plt.xlabel("Threshold")
    plt.ylabel("Rate")
    plt.title(tester[train])
    plt.plot(x, FRR[train], 'red', label="FRR")
    plt.plot(x, FAR[train], 'blue', label="FAR")
    plt.legend()    # 凡例の表示
plt.show()

## ここまで完成 ##

n0 = np.arange(1, len(norm_ave[0])+1)
n1 = np.arange(1, len(norm_ave[1])+1)
n2 = np.arange(1, len(norm_ave[2])+1)
n3 = np.arange(1, len(norm_ave[3])+1)

plt.figure(len(tester))
plt.xlabel("num")
plt.ylabel("norm")
plt.title("vector_norm")
plt.plot(n0, norm_ave[0], 'red', label="fujii")
plt.plot(n1, norm_ave[1], 'blue', label="ooyama")
plt.plot(n2, norm_ave[2], 'yellow', label="okamoto")
plt.plot(n3, norm_ave[3], 'green', label="kajiwara")
plt.legend()    # 凡例の表示
plt.show()
    
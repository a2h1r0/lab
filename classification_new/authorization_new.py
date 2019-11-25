import numpy as np
import pandas as pd
import itertools
from statistics import mean

MIN = 1.5           # 閾値の下限
MAX = 3.0           # 閾値の上限

tester = ["fujii", "ooyama", "matsuda", "kajiwara"]
data = [[0], [0], [0], [0]]
train_size = 2 # 学習する個数

for i in range(len(tester)):
    filename = tester[i] + ".csv"
    data[i] = pd.read_csv(filename, usecols=["in0","in1","in2","in3","in4","in5","in6","in7","in8","in9",
                         "inあ","inい","inう","inA","inB","inC",
                         "in10","in11","in12","in13","in14","in15","in16","in17","in18","in19",
                         "inア","inイ","inウ","inD","inE","inF","Time","Number"], encoding='Shift-JIS')
    data[i].fillna(0, inplace=True)

norm_ave = [[] for i in range(len(tester))]
for order in range(len(tester)):
    num = 0   # データ変更時に初期化   
    norm_sum = 0

    for row in data[order].itertuples(name=None): # 1行ずつ読み出し
        if not (row[-1] in [0,1]):    # 最初以外の区切り文字が出てきたら実行，出てこない間はスキップ
            norm_ave[order].append(norm_sum/num)    # 区切りごとに平均を保存
            num = 0  # カウンタを初期化
            norm_sum = 0    # ベクトル合計を初期化
            
                 
        norm = np.linalg.norm(row[1:34]) # ベクトル距離計算
        norm_sum += norm    # 距離の合計に加算
        num += 1 # 計算回数を増加(データ数の確認)
        
    norm_ave[order].append(norm_sum/num)   # 最終要素には区切り文字がないため．


FRR = np.zeros(len(norm_ave)) # False Rejection Rate(本人拒否率)
FAR = np.zeros(len(norm_ave)) # False Acceptance Rate(他人受入率)

for train in range(len(tester)):    # 1人ごと学習データにする．
    print(tester[train] + "が学習データです．")

    combinations = list(itertools.combinations(range(len(norm_ave[train])), train_size))    # 組み合わせ取得
    FRR_temp = np.zeros(len(combinations))
    FAR_temp = np.zeros(len(combinations))
    for order in range(len(combinations)):     # 組み合わせの個数だけ比較，1人ごとに交差検証
        num = 0

        ##まずは自分と計算
        threshold = 0.1
        for attack in range(len(tester)):  # 1人ごとに認証データにする
            
            for i in range(len(norm_ave[train])):   # 1人のデータ数回比較していく，1データずつ処理
                if (attack == train and i in combinations[order]): # 比較しているのが自分かつ，学習データに用いているデータ同士は比較しない
                    continue
                distance_small = float('inf')
                for j in range(train_size): # 学習データ数1つずつ比較
                    distance = abs(norm_ave[train][i]-norm_ave[train][combinations[order][j]])    # 距離の絶対値
                    if (distance < distance_small):
                        distance_small = distance
                    ## ここで組み合わせの中で距離が小さい方を保存
            
              
                # 距離の差が閾値以下なら，受け入れる
                if (distance_small <= threshold and attack != train):
                    FAR_temp[order] += 1 # 他人受入数
                # 距離が閾値より大きいなら，弾く
                elif (distance_small > threshold and attack == train):
                    FRR_temp[order] += 1 # 本人拒否数
                    
                num += 1    # 判別回数を増加
                
                    
                    
                
        ## 割合計算，組み合わせが変わるごとに計算
        num = num-(len(norm_ave[train])-train_size) # 判別回数から自分と比較した回数を引く，総データ数-学習に使う数が自分の判別回数
        FRR_temp[order] = FRR_temp[order]/(len(norm_ave[train])-train_size)
        FAR_temp[order] = FAR_temp[order]/num
        
    FRR[train] = mean(FRR_temp)*100
    FAR[train] = mean(FAR_temp)*100
        
    # 1人ごとに結果を表示，交差検証の平均
    print("FRR:", end="")
    print(FRR[train]/len(norm_ave[train]))        
    print("FAR:", end="")
    print(FAR[train]/len(norm_ave[train]))
        
        
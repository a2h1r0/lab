import numpy as np
import pandas as pd
import itertools

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
norm_sum = 0
train_num = 0
for order in range(len(tester)):
    for row in data[order].itertuples(name=None): # 1行ずつ読み出し
        if not (row[-1] in [0,1]):    # 最初以外の区切り文字が出てきたら実行，出てこない間はスキップ
            norm_ave[order].append(norm_sum/train_num)    # 区切りごとに平均を保存
            train_num = 0  # カウンタを初期化
            norm_sum = 0    # ベクトル合計を初期化
                 
        norm = np.linalg.norm(row[1:34]) # ベクトル距離計算
        norm_sum += norm    # 距離の合計に加算
        train_num += 1 # 計算回数を増加
        
    norm_ave[order].append(norm_sum/train_num)   # 最終要素には区切り文字がないため．




for order in range(len(tester)):
    print(tester[order] + "が学習データです．")

    conbinations = list(itertools.combinations(range(len(norm_ave[order])), train_size))
    
    for conb_order in range(len(conbinations)):
        conbinations[conb_order] = list(conbinations[conb_order])
        print("組み合わせは ")
        print(np.array(conbinations[conb_order])+np.array([1,1]))
        print(" です．")

        train_data = []        
        for i in range(train_size):
            train_data.append(norm_ave[order][conbinations[conb_order][i]])
            
            ##計算


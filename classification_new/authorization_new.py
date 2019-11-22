import numpy as np
import pandas as pd
import itertools

MIN = 1.5           # 閾値の下限
MAX = 3.0           # 閾値の上限

tester = ["fujii", "ooyama", "matsuda", "kajiwara"]
data = [[0], [0], [0], [0]]
train_size = 2 # 学習する個数

for read in range(len(tester)):
    filename = tester[read] + ".csv"
    data[read] = pd.read_csv(filename, usecols=["in0","in1","in2","in3","in4","in5","in6","in7","in8","in9",
                         "inあ","inい","inう","inA","inB","inC",
                         "in10","in11","in12","in13","in14","in15","in16","in17","in18","in19",
                         "inア","inイ","inウ","inD","inE","inF","Time","Number"], encoding='Shift-JIS')
    data[read].fillna(0, inplace=True)


for order in range(len(tester)):
    print(tester[order] + "が学習データです．")

    conbinations = list(itertools.combinations(range(1, int(data[order]["Number"].max()+1)), train_size))

    for i in range(len(conbinations)):
        print("組み合わせは ")
        print(conbinations[i])
        print(" です．")
        
        norm_sum = 0
        train_num = 0
        train_start = 0
        train_order = -1
        
        for train in data[order].itertuples(name=None): # 1行ずつ読み出し
            separate = train[-1]    # 区切り文字を保存(区切り以外は"0")
            if (separate != 0):    # 区切り文字が出てきたら実行，出てこない間はスキップ
                if (train_order != -1):  # 区切り文字"1"では実行しない
                    norm_ave[train_order] = norm_sum / train_num    # 区切りごとに平均を保存
                    norm_ave.append(separate)
                    train_size = 0  # カウンタを初期化
                train_order += 1    # ノルム平均の保存先を移動(初期状態0番)

                
            norm = np.linalg.norm(train[1:34]) # ベクトル距離計算
            norm_sum += norm    # 距離の合計に加算

            if (train[-1] in conbinations[i]):  # 区切りが組み合わせに含まれる場合
                if (train_start == 1):  # 組み合わせが連番だった場合，更新時に計算結果を保存
                    norm_ave[train_order] = norm_sum / train_num
                    train_size = 0


# =============================================================================
#         norm_ave = [[0] *  for i in range(len(conbinations))]
#         for train in data[order].itertuples(name=None): # 1行ずつ読み出し
#             if (train[-1] != 0):    # 区切り文字が出てきたら実行，出てこない間はスキップ
#                 train_order += 1    # ノルム平均の保存先を移動(初期状態0番)
# 
#                 if (train[-1] in conbinations[i]):  # 区切りが組み合わせに含まれる場合
#                     if (train_start == 1):  # 組み合わせが連番だった場合，更新時に計算結果を保存
#                         norm_ave[train_order] = norm_sum / train_num
#                         train_size = 0
#                         
#                     train_start = 1 # 学習開始
#                     
#                 else:   # 区切りが組み合わせに含まれない場合
#                     if (train_size != 0):   # 学習中であれば
#                         norm_ave[train_order] = norm_sum / train_num   # 平均を保存
#                         train_start = 0 # 学習を停止
#                         train_num = 0  # カウンタを初期化
# 
#             # 区切り文字がない部分，学習中は計算，学習中でなければスキップ
#             if (train_start == 1):
#                 norm = np.linalg.norm(train) # ベクトル距離計算
#                 norm_sum += norm    # 距離の合計に加算
#                 train_num += 1     # 計算回数を増加
#           
#         # ファイル終了時には区切り文字がないので
#         if (train_start != 0):  # 学習中のままファイルが終了した場合
#             norm_ave[train_order] = norm_sum / train_num   # 平均を保存
#                         
# =============================================================================
print(norm_ave)
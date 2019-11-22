import numpy as np
import pandas as pd
import itertools

MIN = 1.5           # 閾値の下限
MAX = 3.0           # 閾値の上限

tester = ["fujii", "ooyama", "matsuda", "kajiwara"]
data = [[0], [0], [0], [0]]
separator = [[0], [0], [0], [0]]

for read in range(len(tester)):
    filename = tester[read] + ".csv"
    data[read] = np.loadtxt(filename, delimiter = ",", dtype = float, skiprows = 1, usecols = range(0,32))   # 認証データ
    separator[read] = pd.read_csv(filename, usecols=['Number'], encoding='Shift-JIS')
    separator[read].fillna(0, inplace=True)


for order in range(len(tester)):
    print(tester[order] + "が学習データです．")

    conbinations = list(itertools.combinations(range(1, int(separator[order].max()+1)), 2))

    for i in range(len(conbinations)):
        print("組み合わせは ")
        print(conbinations[i])
        print(" です．")
        
        norm_sum = 0
        number = -1
        train_size = 0
        train_start = 0
        train_order = -1
        norm_ave = [0, 0]
        for train in data[order]:
            number += 1
            if (int(separator[order].iloc[number]) != zero):
                if (separator[order].iloc[number] in conbinations[i]):
                    train_start = 1
                    train_order += 1
                else:
                    if (train_size != 0):
                        norm_ave[train_order] = norm_sum / train_size
                        train_size = 0
                        train_start = 0
        
            if (train_start == 1):
                norm = np.linalg.norm(train) # ベクトル距離計算
                norm_sum += norm
                train_size += 1
          
        if (train_start != 0):  # ファイル終了時は区切り文字が出てこないため
            norm_ave[train_order] = norm_sum / train_size
                        
print(norm_ave)
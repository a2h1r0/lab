tester = ['ooyama', 'okamoto', 'kajiwara', 'sawano', 'nagamatsu', 'noda', 'hatta', 'fujii', 'matsuda']  # **被験者**



import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../modules'))
from calculate_vector_ave_cols import calculate_vector_ave_cols as cal
from sklearn import svm
from sklearn.model_selection import cross_val_score
import itertools
import copy
import csv


def make_dataset():
    data = []
    label = []
    for item in vector_ave:
        data.extend(item)
    for index, name in enumerate(tester):
        label.extend([name]*len(vector_ave[index]))
    return data, label
    
    


start = 16
to = 17
filename = '16.csv'

cols_full = ['in0','in1','in2','in3','in4','in5','in6','in7',
             'in8','in9','inあ','inい','inう','inA','inB','inC',
             'in10','in11','in12','in13','in14','in15','in16',
             'in17','in18','in19','inア','inイ','inウ','inD','inE',
             'inF']


clf = svm.SVC(C=1.0, kernel='linear')


with open(filename, 'w', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['使用センサ', '結果'])

    for num in range(start, to):
        print(str(len(cols_full)-num)+'個のセンサを使用します．')    
        combinations = list(itertools.combinations(cols_full, num))    # 組み合わせの取得
        print('組み合わせはk='+str(len(combinations))+'通りです．')
        score = [[] for i in range(2)]
        for combination in combinations:
            cols = copy.copy(cols_full)
            for item in combination:
                cols.remove(item)
                
                
            cols.append('Number')
            
            vector_ave = cal(tester, cols)  # ベクトルの平均値を計算
            data, label = make_dataset()
    
            # 組み合わせごとに交差検証した結果を追加
            score[0].append(combination)
            score[1].append(np.mean(cross_val_score(clf, data, label, cv=5)))
            if score[1][-1] == 1.0:
                break
            
        # 組み合わせの中で一番精度の良かった結果を保存
        if combination == ():
            max_cols = ['全部']
        else:
            max_cols = score[0][score[1].index(np.max(score[1]))]
            
        max_score = np.max(score[1])
     
        string = ''
        for s in max_cols:
            string += (s + ', ')
        
        row = [string.rstrip(', '), max_score]
        writer.writerow(row)
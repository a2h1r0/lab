tester = ['ooyama', 'okamoto', 'kajiwara', 'sawano', 'nagamatsu', 'noda', 'hatta', 'fujii', 'matsuda']  # **被験者**
filename = 'output.csv'


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
    
    



cols_full = ['in0','in1','in2','in3','in4','in5','in6','in7',
             'in8','in9','inあ','inい','inう','inA','inB','inC',
             'in10','in11','in12','in13','in14','in15','in16',
             'in17','in18','in19','inア','inイ','inウ','inD','inE',
             'inF']


clf = svm.SVC(C=1.0, kernel='linear')


with open(filename, 'w', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['使用センサ', '結果'])

    for num in range(len(cols_full)):
        print(str(len(cols_full)-num)+'個のセンサを使用します．')    
        combinations = list(itertools.combinations(cols_full, num))    # 組み合わせの取得
        print('組み合わせはk='+str(len(combinations))+'通りです．')
        score = 0
        for index in range(len(combinations)):
            cols = copy.copy(cols_full)
            for item in combinations[0]:
                cols.remove(item)
                
            cols.append('Number')
            
            vector_ave = cal(tester, cols)  # ベクトルの平均値を計算
            data, label = make_dataset()
    
            # 組み合わせの中で一番精度の良かった結果を保存
            temp = np.mean(cross_val_score(clf, data, label, cv=5))
            if temp > score:
                score = temp
                sensors = combinations[0]
            
            if score == 1.0:
                break

            del combinations[0]
            

            
        print('Cross-Validation scores: {}\n'.format(score))
     
        string = ''
        for s in sensors:
            string += (s + ', ')
        
        if string == '':
            string = 'all'

        row = [string.rstrip(', '), score]
        writer.writerow(row)
    
tester = ['ooyama', 'okamoto', 'kajiwara', 'sawano', 'nagamatsu', 'noda', 'hatta', 'fujii', 'matsuda']  # **被験者**



import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../modules'))
from calculate_vector_ave_cols import calculate_vector_ave_cols as cal
from sklearn import svm
from scipy.spatial import distance
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
import itertools
import copy


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

#cols_full = ["in0","in1","in2","in3"]




clf = svm.SVC(C=1.0, kernel='linear')


scores = []
max_cols = []
for num in range(len(cols_full)):
    print(str(num)+'個のセンサを使用します．')    
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
        max_cols.append('32個')
    else:
        max_cols.append(score[0][score[1].index(np.max(score[1]))])
    scores.append(np.max(score[1]))
    print('Cross-Validation scores: {}\n'.format(scores[-1]))

sensor_num = list(range(len(cols_full), 0, -1))

plt.plot(sensor_num, scores, 'red')
plt.xlabel('Sensor Num', fontsize=18)
plt.ylabel('Scores', fontsize=18)
plt.tick_params(labelsize=18)


plt.savefig('sensor.jpg', bbox_inches='tight', pad_inches=0)

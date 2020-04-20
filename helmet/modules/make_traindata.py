import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../modules'))
from calculate_vector_ave import calculate_vector_ave as cal
import numpy as np
import csv

def make_dataset():
    dataset = []
    for index, name in enumerate(tester):
        for data in vector_ave[index]:
            dataset.append(data)
            dataset[-1] = np.append(dataset[-1], name)
    return dataset


tester = ['ooyama', 'okamoto', 'kajiwara', 'sawano', 'nagamatsu',
          'noda', 'hatta', 'fujii', 'matsuda']  # **被験者**
cols = ['in0','in1','in2','in3','in4','in5','in6','in7',
        'in8','in9','inあ','inい','inう','inA','inB','inC',
        'in10','in11','in12','in13','in14','in15','in16', 'in17',
        'in18','in19','inア','inイ','inウ','inD','inE','inF', 'label']


# 平均値計算
vector_ave = cal(tester)

# データ整列
dataset = make_dataset()


with open('train_data.csv', 'w', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(cols)
    
    for row in dataset:
        writer.writerow(row)

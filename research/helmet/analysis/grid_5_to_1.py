tester = ['ooyama', 'okamoto', 'kajiwara', 'sawano', 'nagamatsu', 'noda', 'hatta', 'fujii', 'matsuda']  # **被験者**
filename = '8_to_1.csv'
start = 24
to = 32


import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../modules'))
from calculate_vector_ave_cols import calculate_vector_ave_cols as cal
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import itertools
import copy
import csv


### グリッドサーチで探索するパラメータ空間
def param():
  ret = {
      'C':[1, 10, 100],
      'kernel':['rbf', 'linear', 'poly'],
      'degree':np.arange(1, 6, 1),
      'gamma':np.linspace(0.01, 1.0, 50)
  }
  return ret


def make_dataset(vector_ave):
    data = []
    label = []
    for item in vector_ave:
        data.extend(item)
    for index, name in enumerate(tester):
        label.extend([name]*len(vector_ave[index]))
    return data, label
    
    

def main():
    grid_search = GridSearchCV(SVC(), param(), cv=5)

    cols_full = ['0','1','2','3','4','5','6','7','8','9','あ','い','う','A','B','C','10',
                 '11','12','13','14','15','16','17','18','19','ア','イ','ウ','D','E','F']
        
    
    with open(filename, 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['使用センサ', '結果'])
    
        for num in range(start, to):
            print(str(len(cols_full)-num)+'個のセンサを使用します．')    
            combinations = list(itertools.combinations(cols_full, num))    # 組み合わせの取得
            print('組み合わせはk='+str(len(combinations))+'通りです．')
            score = 0
            for combination in combinations:
                cols = copy.copy(cols_full)
                for item in combination:
                    cols.remove(item)
    
                cols = ['in'+s for s in cols]                
                cols.append('Number')
                
                vector_ave = cal(tester, cols)  # ベクトルの平均値を計算
                data, label = make_dataset(vector_ave)
        
                # 組み合わせの中で一番精度の良かった結果を保存
                grid_search.fit(data, label)
                temp = grid_search.best_score_

                if temp > score:
                    score = temp
                    sensors = combination
                
                if score == 1.0:
                    break
    
                
            print('Score: {}\n'.format(score))
         
            sensors = ['in'+s for s in sensors]
            string = ''
            for s in sensors:
                string += (s + ', ')
            
            if string == '':
                string = 'all'
    
            row = [string.rstrip(', '), score]
            writer.writerow(row)
    
if __name__ == '__main__':
    main()

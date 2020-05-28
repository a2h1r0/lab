tester = ['ooyama', 'okamoto', 'kajiwara', 'sawano', 'nagamatsu', 'noda', 'hatta', 'fujii', 'matsuda']  # **被験者**


import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../modules'))
from calculate_vector_ave_cols import calculate_vector_ave_cols as cal
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV



### グリッドサーチで探索するパラメータ空間
def param():
  ret = {
      'C':[1, 10, 100],
      'kernel':['rbf', 'linear', 'poly'],
      'degree':np.arange(1, 6, 1),
      'gamma':np.linspace(0.01, 1.0, 50)
  }
  return ret


def use_cols(num):
    if num == 5:    
        return ['3','4','C','ウ','D']
    elif num == 4:
        return ['0','3','5','16']
    elif num == 3:
        return ['3','11','E']
    elif num == 2:
        return ['3','F']
    elif num == 1:
        return ['10']


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

    for num in range(5, 0, -1):
        print(str(num)+'個のセンサを使用します．')    

        cols = ['in'+s for s in use_cols(num)]
        cols.append('Number')
        
        vector_ave = cal(tester, cols)
        data, label = make_dataset(vector_ave)

        grid_search.fit(data, label)

        print(grid_search.best_score_)
        print(grid_search.best_params_) 
        print('\n----------------\n\n')



if __name__ == '__main__':
    main()
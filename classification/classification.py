tester = ["ooyama", "okamoto", "kajiwara", "sawano", "nagamatsu", "noda", "hatta", "fujii", "matsuda"]  # **被験者**




import numpy as np
import calculate_vector_ave as cal
from sklearn import svm
from scipy.spatial import distance
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate


vector_ave = cal.calculate_vector_ave(tester)  # ベクトルの平均値を計算
data = []
for item in vector_ave:
    data.extend(item)

label = []
for index, name in enumerate(tester):
    label.extend([name]*len(vector_ave[index]))


clf = svm.SVC(C=1.0, kernel='linear')
scores = cross_val_score(clf, data, label, cv=5)

print('Cross-Validation scores: {}'.format(scores))
print('Average score: {}'.format(np.mean(scores)))

"""
print (classification_report(test_label, test_pred))
print (accuracy_score(test_label, test_pred))
print (confusion_matrix(test_label, test_pred))
"""



#plt.savefig("EER.eps", bbox_inches='tight', pad_inches=0)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA


# left_hip
train_data = np.loadtxt("left_hip_train.csv", delimiter=",", skiprows=1)
test_data = np.loadtxt("left_hip_test.csv", delimiter=",", skiprows=1)
# X平均, Y平均, Z平均, X分散, Y分散, Z分散にする．
left_hip_train = []
data_size = int(len(train_data)/10)
if data_size != 0:
    for i in range(10, len(train_data), data_size):
        left_hip_train.append(np.concatenate([np.mean(train_data[i-10:i], axis=0), 
                                    np.var(train_data[i-10:i], axis=0)]))
left_hip_test = []
data_size = int(len(test_data)/10)
if data_size != 0:
    for i in range(10, len(test_data), data_size):
        left_hip_test.append(np.concatenate([np.mean(test_data[i-10:i], axis=0), 
                                    np.var(test_data[i-10:i], axis=0)]))

    
    
# right_arm
train_data = np.loadtxt("right_arm_train.csv", delimiter=",", skiprows=1)
test_data = np.loadtxt("right_arm_test.csv", delimiter=",", skiprows=1)
# X平均, Y平均, Z平均, X分散, Y分散, Z分散にする．
right_arm_train = []
data_size = int(len(train_data)/10)
if data_size != 0:
    for i in range(10, len(train_data), data_size):
        right_arm_train.append(np.concatenate([np.mean(train_data[i-10:i], axis=0), 
                                    np.var(train_data[i-10:i], axis=0)]))
right_arm_test = []
data_size = int(len(test_data)/10)
if data_size != 0:
    for i in range(10, len(test_data), data_size):
        right_arm_test.append(np.concatenate([np.mean(test_data[i-10:i], axis=0), 
                                    np.var(test_data[i-10:i], axis=0)]))



# left_wrist
train_data = np.loadtxt("left_wrist_train.csv", delimiter=",", skiprows=1)
test_data = np.loadtxt("left_wrist_test.csv", delimiter=",", skiprows=1)
# X平均, Y平均, Z平均, X分散, Y分散, Z分散にする．
left_wrist_train = []
data_size = int(len(train_data)/10)
if data_size != 0:
    for i in range(10, len(train_data), data_size):
        left_wrist_train.append(np.concatenate([np.mean(train_data[i-10:i], axis=0), 
                                    np.var(train_data[i-10:i], axis=0)]))
left_wrist_test = []
data_size = int(len(test_data)/10)
if data_size != 0:
    for i in range(10, len(test_data), data_size):
        left_wrist_test.append(np.concatenate([np.mean(test_data[i-10:i], axis=0), 
                                    np.var(test_data[i-10:i], axis=0)]))



# right_wrist
train_data = np.loadtxt("right_wrist_train.csv", delimiter=",", skiprows=1)
test_data = np.loadtxt("right_wrist_test.csv", delimiter=",", skiprows=1)
# X平均, Y平均, Z平均, X分散, Y分散, Z分散にする．
right_wrist_train = []
data_size = int(len(train_data)/10)
if data_size != 0:
    for i in range(10, len(train_data), data_size):
        right_wrist_train.append(np.concatenate([np.mean(train_data[i-10:i], axis=0), 
                                    np.var(train_data[i-10:i], axis=0)]))
right_wrist_test = []
data_size = int(len(test_data)/10)
if data_size != 0:
    for i in range(10, len(test_data), data_size):
        right_wrist_test.append(np.concatenate([np.mean(test_data[i-10:i], axis=0), 
                                    np.var(test_data[i-10:i], axis=0)]))



    
train = np.concatenate([left_hip_train, right_arm_train], axis=1)
test = np.concatenate([left_hip_test, right_arm_test], axis=1)


    
    
    
    
    
    
    
"""
生データの可視化
x_axis_train = list(range(len(train_data)))
plt.plot(x_axis_train, train_data)    
x_axis_test = list(range(len(test_data)))
plt.plot(x_axis_test, test_data)    
"""



pca = PCA(n_components=2)
pca.fit(train)
pca_train = pca.components_
pca.fit(test)
pca_test = pca.components_



x_axis = list(range(len(train)))
plt.scatter(x_axis, pca_train)
plt.scatter(x_axis, pca_test)   

    

"""
OneClassSVM：ダメダメ

clf = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(train)
pred = clf.predict(test)
print(pred)
"""    


"""
LOF：ダメダメ

model = LocalOutlierFactor(n_neighbors=1,
                           novelty=True,
                           contamination=0.1)
model.fit(train) # train_dataは正常データが大多数であるような訓練データ
prediction = model.predict(test) # テストデータに対する予測

print(prediction)
"""
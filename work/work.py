import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from scipy import signal
from numpy.fft import fft, ifft, fftfreq
from sklearn.cluster import KMeans





# left_hip
take_data = np.loadtxt("left_hip_train.csv", delimiter=",", skiprows=1)
take_wash_data = np.loadtxt("left_hip_test.csv", delimiter=",", skiprows=1)

"""
# 合成波
take = []
take_wash = []
for item in take_data:
    take.append(np.linalg.norm(item))
for item in take_wash_data:
    take_wash.append(np.linalg.norm(item))

take_fft = fft(take)
take_wash_fft = fft(take_wash)

wash_fft = take_wash_fft - take_fft
#wash = ifft(wash_fft)


take_wash2_data = np.loadtxt("subject1_file_523.csv", delimiter=",", skiprows=1)

take_wash2 = []
for item in take_wash2_data[:len(take_data)]:
    take_wash2.append(np.linalg.norm(item))

take_wash2_fft = fft(take_wash2)

take2_fft = take_wash2_fft - wash_fft
take2 = ifft(take2_fft)
"""


take_x = []
take_y = []
take_z = []
for val in take_data:
    take_x.append(val[0])
    take_y.append(val[1])
    take_z.append(val[2])
take_x_fft = fft(take_x)
take_y_fft = fft(take_y)
take_z_fft = fft(take_z)
    
take_wash_x = []
take_wash_y = []
take_wash_z = []
for val in take_wash_data[:len(take_data)]:
    take_wash_x.append(val[0])
    take_wash_y.append(val[1])
    take_wash_z.append(val[2])

take_wash_x_fft = fft(take_wash_x)
take_wash_y_fft = fft(take_wash_y)
take_wash_z_fft = fft(take_wash_z)


wash_x_fft = take_wash_x_fft - take_x_fft
wash_y_fft = take_wash_y_fft - take_y_fft
wash_z_fft = take_wash_z_fft - take_z_fft


take_wash2_data = np.loadtxt("subject1_file_523.csv", delimiter=",", skiprows=1)
take_wash2_x = []
take_wash2_y = []
take_wash2_z = []
for val in take_wash2_data[:len(take_data)]:
    take_wash2_x.append(val[0])
    take_wash2_y.append(val[1])
    take_wash2_z.append(val[2])
take_wash2_x_fft = fft(take_wash2_x)
take_wash2_y_fft = fft(take_wash2_y)
take_wash2_z_fft = fft(take_wash2_z)



take2_x_fft = take_wash2_x_fft - wash_x_fft
take2_y_fft = take_wash2_y_fft - wash_y_fft
take2_z_fft = take_wash2_z_fft - wash_z_fft

take2_x = ifft(take2_x_fft)
take2_y = ifft(take2_y_fft)
take2_z = ifft(take2_z_fft)
take2_x = take2_x.real
take2_y = take2_y.real
take2_z = take2_z.real


take2_data = np.c_[take2_x, take2_y, take2_z]






"""
freq = fftfreq(len(sum1), d = 1/(len(sum1)/30)) 
plt.plot(freq[1:int(len(sum1)/2)], abs(sum_fft[1:int(len(sum1)/2)]))
plt.yscale('log')
plt.xlabel('Freq Hz')
plt.ylabel('Power')
"""



"""
x = []
y = []
z = []
for val in train_data:
    x.append(val[0])
    y.append(val[1])
    z.append(val[2])
x2 = []
y2 = []
z2 = []
for val in train_data:
    x2.append(val[0])
    y2.append(val[1])
    z2.append(val[2])

sum1 = x + y + z
fft1 = fftn(sum1)
freq = fftfreq(len(sum1), d = 1/(len(sum1)/30))  
fig, axes = plt.subplots(figsize=(10, 5), ncols=4, sharey=True)
ax = axes[0]
ax.plot(freq[1:int(len(sum1)/2)], abs(fft1[1:int(len(sum1)/2)]))
ax.set_yscale('log')
ax.set_xlabel('Freq Hz')
ax.set_ylabel('Power')
# 周波数 f → 周期 T に直して表示する
# 周期は fT = 1 を満たすので単に逆数にすれば良い
ax = axes[1]
ax.plot(1 / freq[1:int(len(sum1)/2)], abs(fft1[1:int(len(sum1)/2)]))
ax.set_yscale('log')
ax.set_xlabel('T s')
ax.set_xscale('log')

sum2 = x2 + y2 + z2
fft2 = fftn(sum2)
freq2 = fftfreq(len(sum2), d = 1/(len(sum2)/30))   
ax = axes[2]
ax.plot(freq2[1:int(len(sum2)/2)], abs(fft2[1:int(len(sum2)/2)]))
ax.set_yscale('log')
ax.set_xlabel('Freq Hz')
ax.set_ylabel('Power')
# 周波数 f → 周期 T に直して表示する
# 周期は fT = 1 を満たすので単に逆数にすれば良い
ax = axes[3]
ax.plot(1 / freq2[1:int(len(sum2)/2)], abs(fft2[1:int(len(sum2)/2)]))
ax.set_yscale('log')
ax.set_xlabel('T s')
ax.set_xscale('log')
"""















"""
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
"""

      
"""
K-Means

kmeans_model = KMeans(n_clusters=2, random_state=10).fit(left_hip_test)
labels = kmeans_model.labels_
print(labels)
"""



"""
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

"""
train_data = np.loadtxt("left_hip_train.csv", delimiter=",", skiprows=1)
test_data = np.loadtxt("left_hip_test.csv", delimiter=",", skiprows=1)

x = []
y = []
z = []
for val in train_data:
    x.append(val[0])
    y.append(val[1])
    z.append(val[2])
"""

"""
# FFT Scipy

x_f = np.fft.fft(x)
y_f = np.fft.fft(y)
z_f = np.fft.fft(z)

x_abs = np.abs(x_f)
#y_abs = np.abs(y_f)
#z_abs = np.abs(z_f)
#plt.plot(x_abs)    
#plt.plot(y_abs)
#plt.plot(z_abs)

x2 = []
y2 = []
z2 = []
for val in test_data:
    x2.append(val[0])
    y2.append(val[1])
    z2.append(val[2])

x2_f = np.fft.fft(x2)
y2_f = np.fft.fft(y2)
z2_f = np.fft.fft(z2)

x2_abs = np.abs(x2_f)
#y2_abs = np.abs(y2_f)
#z2_abs = np.abs(z2_f)
#plt.plot(x2_abs)    
#plt.plot(y2_abs)
#plt.plot(z2_abs)
"""

"""
# FFT np こっちが良い
x_fft = fftn(x)
freq = fftfreq(len(x), d = 1/(len(x)/30))  
fig, axes = plt.subplots(figsize=(10, 5), ncols=4, sharey=True)
ax = axes[0]
ax.plot(freq[1:int(len(x)/2)], abs(x_fft[1:int(len(x)/2)]))
ax.set_yscale('log')
ax.set_xlabel('Freq Hz')
ax.set_ylabel('Power')
# 周波数 f → 周期 T に直して表示する
# 周期は fT = 1 を満たすので単に逆数にすれば良い
ax = axes[1]
ax.plot(1 / freq[1:int(len(x)/2)], abs(x_fft[1:int(len(x)/2)]))
ax.set_yscale('log')
ax.set_xlabel('T s')
ax.set_xscale('log')

x2_fft = fftn(x2)
freq2 = fftfreq(len(x2), d = 1/(len(x2)/30))   
ax = axes[2]
ax.plot(freq2[1:int(len(x2)/2)], abs(x2_fft[1:int(len(x2)/2)]))
ax.set_yscale('log')
ax.set_xlabel('Freq Hz')
ax.set_ylabel('Power')
# 周波数 f → 周期 T に直して表示する
# 周期は fT = 1 を満たすので単に逆数にすれば良い
ax = axes[3]
ax.plot(1 / freq2[1:int(len(x2)/2)], abs(x2_fft[1:int(len(x2)/2)]))
ax.set_yscale('log')
ax.set_xlabel('T s')
ax.set_xscale('log')
"""

    
    
    
"""
# 生データの可視化
x_axis_train = list(range(len(train_data)))
plt.plot(x_axis_train, train_data)    
x_axis_test = list(range(len(test_data)))
plt.plot(x_axis_test, test_data)    
"""


"""
# PCA：いらんかった

pca = PCA(n_components=2)
pca.fit(train)
pca_train = pca.components_
pca.fit(test)
pca_test = pca.components_

x_axis = list(range(len(train)))
plt.scatter(x_axis, pca_train)
plt.scatter(x_axis, pca_test)   
"""
    

"""
# OneClassSVM：ダメダメ

clf = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(train)
pred = clf.predict(test)
print(pred)
"""    


"""
# LOF：ダメダメ

model = LocalOutlierFactor(n_neighbors=1,
                           novelty=True,
                           contamination=0.1)
model.fit(train) # train_dataは正常データが大多数であるような訓練データ
prediction = model.predict(test) # テストデータに対する予測
print(prediction)
"""
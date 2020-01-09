tester = ["ooyama", "okamoto", "kajiwara", "sawano", "nagamatsu", "noda", "hatta", "fujii", "matsuda"]  # **被験者**
subject = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]  # **プロット名**
marker = ["o", ",", "^", "*", "v", "1", "p", "D", "x"]   # **プロットマーカー**
#################



import calculate_vector_ave as cal
from sklearn import decomposition
import matplotlib.pyplot as plt

# ベクトルの平均値を計算
vector_ave = cal.calculate_vector_ave(tester)

# testにベクトルの平均値をまとめる
test = []
for name in range(len(tester)):
    test.extend(vector_ave[name])

# 計算
model = decomposition.PCA(n_components=2)
model.fit(test)
compressed = model.transform(test)

# 描画
plt.xlabel("First component", fontsize=14)
plt.ylabel("Second component", fontsize=14)
plt.tick_params(labelsize=14)
long = 0    # カウンタ
for num, item, name in zip(range(len(tester)), marker, subject):    # 1人ずつ描画
    plt.scatter(compressed[long:long+len(vector_ave[num]), 0],
                compressed[long:long+len(vector_ave[num]), 1], marker=item, label="Subject "+name)
    long += len(vector_ave[num])
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=10)    # 凡例を枠外に
plt.subplots_adjust(right=0.78) # 調整
plt.show()
plt.savefig("PCA.svg")
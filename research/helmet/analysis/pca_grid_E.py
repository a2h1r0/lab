tester = ["nagamatsu"]  # **被験者**
subject = ["E"]  # **プロット名**
marker = ["v"]   # **プロットマーカー**
#################



from sklearn import decomposition
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../modules"))
from calculate_vector_ave import calculate_vector_ave as cal


# ベクトルの平均値を計算
vector_ave = cal(tester)

# testにベクトルの平均値をまとめる
test = []
for name in range(len(tester)):
    test.extend(vector_ave[name])

# 計算
model = decomposition.PCA(n_components=2)
model.fit(test)
compressed = model.transform(test)

# 描画
plt.figure(figsize=(10,8))
plt.xlabel("First component", fontsize=18)
plt.ylabel("Second component", fontsize=18)
plt.tick_params(labelsize=18)
long = 0    # カウンタ
for num, item, name in zip(range(len(tester)), marker, subject):    # 1人ずつ描画
    plt.scatter(compressed[long:long+len(vector_ave[num]), 0],
                compressed[long:long+len(vector_ave[num]), 1], marker=item, label="Subject "+name)
    long += len(vector_ave[num])
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=18)    # 凡例を枠外に
plt.subplots_adjust(right=0.75) # 調整
plt.show()
#plt.savefig("PCA.eps", bbox_inches='tight', pad_inches=0)
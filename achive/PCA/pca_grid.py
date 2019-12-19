import pandas as pd
from sklearn import decomposition
import matplotlib.pyplot as plt

test = pd.read_csv('testdata.csv', usecols=["in0","in1","in2","in3","in4","in5","in6","in7",
                                            "in8","in9","inあ","inい","inう","inA","inB","inC",
                                            "in10","in11","in12","in13","in14","in15","in16",
                                            "in17","in18","in19","inア","inイ","inウ","inD","inE",
                                            "inF"], encoding='Shift-JIS')

model = decomposition.PCA(n_components=2)
model.fit(test)

compressed = model.transform(test)

plt.xlabel("First component", fontsize=14)
plt.ylabel("Second component", fontsize=14)
plt.tick_params(labelsize=14)
plt.scatter(compressed[:20, 0],compressed[:20, 1], marker="o", label="Subject A")
plt.scatter(compressed[20:40, 0],compressed[20:40, 1], marker=",", label="Subject B")
plt.scatter(compressed[40:60, 0],compressed[40:60, 1], marker="^", label="Subject C")
plt.scatter(compressed[60:80, 0],compressed[60:80, 1], marker="*", label="Subject D")
plt.scatter(compressed[80:100, 0],compressed[80:100, 1], marker="o", label="Subject E")
plt.scatter(compressed[100:120, 0],compressed[100:120, 1], marker=",", label="Subject F")
plt.scatter(compressed[120:140, 0],compressed[120:140, 1], marker="^", label="Subject G")
plt.scatter(compressed[140:160, 0],compressed[140:160, 1], marker="*", label="Subject H")
plt.legend()
plt.show()
plt.savefig("PCA.svg")
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

plt.scatter(compressed[:20, 0],compressed[:20, 1], marker="o", label="ooyama")
plt.scatter(compressed[20:38, 0],compressed[20:38, 1], marker=",", label="okamoto")
plt.scatter(compressed[38:54, 0],compressed[38:54, 1], marker="^", label="kajiwara")
plt.scatter(compressed[54:74, 0],compressed[54:74, 1], marker="*", label="fujii")
plt.scatter(compressed[74:88, 0],compressed[74:88, 1], marker="D", label="matsuda")
plt.legend()
plt.show()
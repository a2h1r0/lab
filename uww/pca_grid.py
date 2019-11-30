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
plt.scatter(compressed[20:36, 0],compressed[20:36, 1], marker=",", label="okamoto")
plt.scatter(compressed[36:52, 0],compressed[36:52, 1], marker="^", label="kajiwara")
plt.scatter(compressed[52:72, 0],compressed[52:72, 1], marker="*", label="fujii")
plt.scatter(compressed[72:84, 0],compressed[72:84, 1], marker="D", label="matsuda")
plt.legend()
plt.show()
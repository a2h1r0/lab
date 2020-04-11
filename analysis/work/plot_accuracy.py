filename = 'output_half_helmet.txt'



import numpy as np
import matplotlib.pyplot as plt

accuracy = np.loadtxt(filename, delimiter='\t', skiprows=1, usecols=[1])

x = list(range(len(accuracy), 0, -1))





## 結果の描画 ##
plt.figure(0)  # 複数ウィンドウで表示
plt.title("Change of accuracy", fontsize=18)
plt.plot(x, accuracy, 'red')
plt.xlabel("Sensor Num", fontsize=18)
plt.ylabel("Accuracy", fontsize=18)
plt.tick_params(labelsize=18)


plt.savefig("20.eps", bbox_inches='tight', pad_inches=0)

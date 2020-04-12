filename_full = 'output_full_helmet.txt'
filename_half = 'output_half_helmet.txt'



import numpy as np
import matplotlib.pyplot as plt


accuracy_full = np.loadtxt(filename_full, delimiter='\t', skiprows=1, usecols=[1])
x_full = list(range(len(accuracy_full), 0, -1))

accuracy_half = np.loadtxt(filename_half, delimiter='\t', skiprows=1, usecols=[1])
x_half = list(range(len(accuracy_half), 0, -1))




## 結果の描画 ##
plt.figure(0)  # 複数ウィンドウで表示
plt.title("Change of accuracy", fontsize=18)
plt.plot(x_full, accuracy_full, 'red', label="Full helmet")
plt.plot(x_half, accuracy_half, 'blue', linestyle="dashed", label="Half helmet")
plt.xlabel("Sensor Num", fontsize=18)
plt.ylabel("Accuracy", fontsize=18)
plt.tick_params(labelsize=18)
plt.legend(fontsize=18, loc='lower right')  # 凡例の表示


plt.savefig("Acc.eps", bbox_inches='tight', pad_inches=0)

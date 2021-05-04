import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
os.chdir(os.path.dirname(__file__))


# sin波（0 ~ 2）の生成
sin = np.sin(np.linspace(0, 2 * np.pi, 20)) + 1
# 1以上の値を1にする（0 ~ 1）
sin[sin > 1] = 1

# グレースケールへ変換
colors = np.array(sin * 30 + 225, dtype=int)


plt.figure(figsize=(16, 9))
plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
plt.plot(range(len(colors)), colors)
plt.xlabel('Index', fontsize=18)
plt.ylabel('Glay Scale', fontsize=18)
plt.title('Colors', fontsize=18)
plt.tick_params(labelsize=18)

# plt.savefig('../figure/colors_wave.eps', bbox_inches='tight', pad_inches=0)

plt.show()

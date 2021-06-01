from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir(os.path.dirname(__file__))


SOUND_DATA = './sounds/washer.mp3'  # 音源


# ファイルの読み出し
sound = AudioSegment.from_file(SOUND_DATA, 'mp3')

# データの整形
data = np.array(sound.get_array_of_samples())
sample_num = len(data)


plt.figure(figsize=(16, 9))
plt.plot(range(sample_num), data, 'blue')
plt.xlabel('Time [s]', fontsize=18)
plt.ylabel('Sound', fontsize=18)
plt.tick_params(labelsize=18)
# plt.savefig('../figure/power.png', bbox_inches='tight', pad_inches=0)

plt.show()

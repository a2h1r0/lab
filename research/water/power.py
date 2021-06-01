from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir(os.path.dirname(__file__))


SOUND_FILE = 'shampoo_1.mp3'  # 音源


SOUND_DIR = './sounds/'


# ファイルの読み出し
sound = AudioSegment.from_file(SOUND_DIR + SOUND_FILE, 'mp3')

plt.xlabel('Frequency [Hz]', fontsize=18)
plt.ylabel('Amplitude', fontsize=18)
plt.tick_params(labelsize=18)

# データの整形
data = np.array(sound.get_array_of_samples())
data = data[-(WINDOW*1000):]
sample_num = len(data)
sampling_rate = sound.frame_rate

# 周波数の取得
freqs = np.fft.fftfreq(sample_num, d=1.0/sampling_rate)
index = np.argsort(freqs)
# 振幅スペクトル
fft = np.abs(np.fft.fft(data))
# パワースペクトル
power = fft ** 2


plt.figure(figsize=(16, 9))
plt.plot(freqs[index], power[index], 'blue')
plt.xlim(0, max(freqs))
# plt.savefig('../figure/power.png', bbox_inches='tight', pad_inches=0)

plt.show()

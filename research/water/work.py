from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir(os.path.dirname(__file__))


SOUND_DATA = './sounds/test.mp3'  # 音源


# ファイルの読み出し
sound = AudioSegment.from_file(SOUND_DATA, 'mp3')

# データの整形
data = np.array(sound.get_array_of_samples())
sample_num = len(data)
sampling_rate = sound.frame_rate

# 周波数の取得
freqs = np.fft.fftfreq(sample_num, d=1.0/sampling_rate)
# FFT
fft = np.abs(np.fft.fft(data))


plt.figure(figsize=(16, 9))
plt.plot(freqs, fft, 'blue')
plt.xlabel('Frequency [Hz]', fontsize=18)
plt.ylabel('Amplitude', fontsize=18)
plt.tick_params(labelsize=18)
plt.xlim(0, 8000)
plt.ticklabel_format(style='plain', axis='y')
# plt.savefig('../figure/power.png', bbox_inches='tight', pad_inches=0)

plt.show()

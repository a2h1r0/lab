from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir(os.path.dirname(__file__))


SOUND_FILE = 'shampoo_1.mp3'  # 音源
STEP = 3


SOUND_DIR = './sounds/trimmed/48000/'


# ファイルの読み出し
sound = AudioSegment.from_file(SOUND_DIR + SOUND_FILE, 'mp3')

plt.xlabel('Frequency [Hz]', fontsize=18)
plt.ylabel('Amplitude', fontsize=18)
plt.tick_params(labelsize=18)


sound_len = sound.duration_seconds
for i in range(int(sound_len // STEP)):
    start = i * STEP
    if i == (int(sound_len // STEP) - 1):
        show_sound = sound[start:]
    else:
        end = (i + 1) * STEP
        show_sound = sound[start:end]

    data = np.array(show_sound.get_array_of_samples())
    sample_num = len(data)
    sampling_rate = sound.frame_rate

    # 周波数の取得
    freqs = np.fft.fftfreq(sample_num, d=1.0/sampling_rate)
    index = np.argsort(freqs)
    # 振幅スペクトル
    fft = np.abs(np.fft.fft(data))
    # パワースペクトル
    power = fft ** 2

    plt.figure(i, figsize=(16, 9))
    plt.title(str(start) + 's to ' + str(start + STEP) + 's')
    plt.plot(freqs[index], power[index], 'blue')
    plt.xlim(0, max(freqs))


# plt.savefig('../figure/power.png', bbox_inches='tight', pad_inches=0)
plt.show()

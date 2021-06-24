from pydub import AudioSegment
from pydub.silence import split_on_silence
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir(os.path.dirname(__file__))


SOUND_DIR = './sounds/'
SOUND_FILE = 'shampoo_1.mp3'  # 音源


# ファイルの読み出し
sound = AudioSegment.from_file(SOUND_DIR + 'raw/' + SOUND_FILE, 'mp3')


plt.xlabel('Time [s]', fontsize=18)
plt.ylabel('Sound', fontsize=18)
plt.tick_params(labelsize=18)

# データの整形
data = np.array(sound.get_array_of_samples())
sample_num = len(data)

plt.figure(figsize=(16, 9))
plt.title('Overview')
plt.plot(range(sample_num), data)

# 分割
chunks = split_on_silence(sound, min_silence_len=2000,
                          silence_thresh=-55)
for index, chunk in enumerate(chunks):
    # データの整形
    data = np.array(chunk.get_array_of_samples())
    sample_num = len(data)
    plt.figure(index, figsize=(16, 9))
    plt.title('Split')
    plt.plot(range(sample_num), data)

plt.show()


# 切り出し部分の保存
chunks[0].export(SOUND_DIR + SOUND_FILE, format="mp3")

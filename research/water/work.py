from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir(os.path.dirname(__file__))

sound_file = './sounds/test.mp3'

sound = AudioSegment.from_file(sound_file, 'mp3')
data = np.array(sound.get_array_of_samples())
# spec = np.fft.fft(data)  # 2次元配列(実部，虚部)
# freq = np.fft.fftfreq(data.shape[0], 1.0/sound.frame_rate)
# spec = spec[:int(spec.shape[0]/2 + 1)]  # 周波数がマイナスになるスペクトル要素の削除
# freq = freq[:int(freq.shape[0]/2 + 1)]  # 周波数がマイナスになる周波数要素の削除
# max_spec = max(np.abs(spec))  # 最大音圧を取得(音圧を正規化するために使用）
# plt.plot(freq, np.abs(spec)/max_spec)

# plt.grid()
# plt.xlim([0, 4000])  # グラフに出力する周波数の範囲[Hz]
# plt.xlabel("Frequency[Hz]")
# plt.ylabel("Sound Pressure[-]")
# plt.yscale("log")
# plt.savefig(file + ".png")  # pngファイルで出力


# データのパラメータ
N = 256            # サンプル数
dt = 0.01          # サンプリング間隔
f1, f2 = 10, 20    # 周波数
# t = np.arange(0, N*dt, dt)  # 時間軸
freq = np.linspace(0, 1.0/dt, N)  # 周波数軸

# 信号を生成（周波数10の正弦波+周波数20の正弦波+ランダムノイズ）
# f = np.sin(2*np.pi*f1*t) + np.sin(2*np.pi*f2*t) + 0.3 * np.random.randn(N)

# 高速フーリエ変換
F = np.fft.fft(data)
rate = sound.frame_rate
time = sound.duration_seconds
t = np.arange(0, time, rate)  # 時間軸

# 振幅スペクトルを計算
Amp = np.abs(F)

# パワースペクトルの計算（振幅スペクトルの二乗）
Pow = Amp ** 2

# グラフ表示
plt.figure()
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 17
plt.subplot(121)
# plt.plot(rate, f, label='f(n)')
# plt.xlabel("Time", fontsize=20)
# plt.ylabel("Signal", fontsize=20)
plt.grid()
leg = plt.legend(loc=1, fontsize=25)
leg.get_frame().set_alpha(1)
plt.subplot(122)
plt.plot(freq, Pow, label='|F(k)|')
plt.xlabel('Frequency', fontsize=20)
plt.ylabel('Amplitude', fontsize=20)
plt.grid()
leg = plt.legend(loc=1, fontsize=25)
leg.get_frame().set_alpha(1)
plt.show()

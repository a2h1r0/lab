from pydub import AudioSegment
import argparse
import urllib.request
import pathlib
import numpy as np
import librosa
import matplotlib.pyplot as plt
import scipy
import os
os.chdir(os.path.dirname(__file__))


SOUND_DATA = '../sounds/temp/shampoo/1.mp3'


def utterance(x, threshold):
    for i, xx in enumerate(x):
        if xx > threshold:
            begin_x = i
            break
    for i, xx in enumerate(x[::-1]):
        if xx > threshold:
            end_x = len(x) - i
            break
    return x[begin_x:end_x]


def main():
    # 音源の読み出し
    sound = AudioSegment.from_file(SOUND_DATA, 'mp3')
    fs = len(sound[:1000].get_array_of_samples())
    x = np.array(sound.get_array_of_samples())

    filters_count = 20
    mfcc_dim = 12

    hanning_x = np.hanning(len(x)) * x
    fft = np.fft.fft(hanning_x)
    amplitude_spectrum = np.abs(fft)
    amplitude_spectrum = amplitude_spectrum[:len(x)//2]
    mel_filter_bank = librosa.filters.mel(sr=fs, n_fft=len(x) - 1, n_mels=filters_count, fmax=fs // 2)
    mel_amplitude_spectrum = np.dot(mel_filter_bank, amplitude_spectrum)
    mel_log_power_spectrum = 20 * np.log10(mel_amplitude_spectrum)
    mfcc = scipy.fftpack.dct(mel_log_power_spectrum, norm='ortho')
    mfcc = mfcc[:mfcc_dim]
    mel_log_power_spectrum_envelope = scipy.fftpack.idct(mfcc, n=filters_count, norm='ortho')

    log_power_spectrum = 20 * np.log10(amplitude_spectrum)
    frequencies = librosa.fft_frequencies(sr=fs, n_fft=len(x) - 1)
    mel_frequencies = librosa.mel_frequencies(n_mels=filters_count+2, fmin=0.0, fmax=fs/2, htk=True)
    mel_frequencies = mel_frequencies[1:-1]

    fig = plt.figure(figsize=(6.4, 4.8))

    ax = fig.add_subplot(2, 1, 1)
    for i in range(filters_count):
        ax.plot(frequencies, mel_filter_bank[i])
    ax.set_xlim(0, fs / 2)
    ax.set_xlabel('frequency [Hz]')
    ax.grid(color='black', linestyle='dotted')
    ax.set_title(f'mel filter bank (num filters: {filters_count})')

    ax = fig.add_subplot(2, 1, 2)
    ax.plot(frequencies, log_power_spectrum, linewidth=1, label='log power spectrum')
    ax.plot(mel_frequencies, mel_log_power_spectrum, linewidth=1, marker='.', label=f'mel log power spectrum (n: {filters_count})')
    ax.plot(mel_frequencies, mel_log_power_spectrum_envelope, linewidth=1, label=f'mfcc log power spectrum (d: {mfcc_dim})')
    ax.legend()
    ax.set_xlim(0, fs / 2)
    # ax.set_ylim(-60, 40)
    ax.set_xlabel('frequency [Hz]')
    ax.set_ylabel('log power [dB]')
    # ax.set_title(f'log power spectrum ({arguments.save_file_name} )')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

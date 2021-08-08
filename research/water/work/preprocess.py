from pydub import AudioSegment
from pydub.silence import split_on_silence
import glob
import os
os.chdir(os.path.dirname(__file__))


SOUND_DIR = '../sounds/raw/'  # 音源データ
TEMP_DIR = '../sounds/temp/'  # 保存先
THRESHOLD = -55  # 無音検出閾値


def preprocess():
    """
    音源データ前処理（無音部分の削除）
    """

    if os.path.exists(TEMP_DIR) == False:
        os.makedirs(TEMP_DIR)

    files = glob.glob(SOUND_DIR + '*.m4a')
    for filename in files:
        sound = AudioSegment.from_file(filename, 'm4a')
        sounds = split_on_silence(sound, min_silence_len=2000, silence_thresh=THRESHOLD)
        sounds[0].export(TEMP_DIR + filename.split('\\')[1].split('.')[0] + '.mp3', format='mp3')


if __name__ == '__main__':
    preprocess()

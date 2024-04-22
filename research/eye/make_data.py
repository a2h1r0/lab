import numpy as np
import pandas as pd
import math
import csv
import glob
import re
import datetime
import random
import copy
import sys
import os
os.chdir(os.path.dirname(__file__))


WINDOW_SIZE = 10  # ウィンドウサイズ（秒）
STEP = 3  # ステップ幅（秒）
USE_COLUMNS = [
    'Recording timestamp', 'Recording date', 'Gaze point X', 'Gaze point Y', 'Gaze point left X', 'Gaze point left Y', 'Gaze point right X', 'Gaze point right Y', 'Gaze direction left X', 'Gaze direction left Y', 'Gaze direction left Z', 'Gaze direction right X', 'Gaze direction right Y', 'Gaze direction right Z', 'Pupil diameter left', 'Pupil diameter right'
]   # 使用カラム


RAW_DIR = './data/raw'
SAVE_DIR = f'./data/preprocess/window_{WINDOW_SIZE}'


def preprocess(filename):
    """
    データの前処理

    Args:
        filename (string): ファイル名
    Returns:
        list: 前処理データ
    """

    def make_label(string):
        """
        ラベルの生成

        Args:
            string (string): 識別情報
        Returns:
            string: ラベル
        """

        label = ''

        if 'drunk' in string:
            label = 'drunk'
        else:
            label = 'sober'

        return label

    def slide_window(raw, label):
        """
        スライディングウィンドウ

        Args:
            raw (list): ローデータ
            label (list): ラベル
        Returns:
            list: スライディングウィンドウ後のデータ
        """

        data = []

        next_start_time = None
        for start_index, start in raw.iterrows():
            start_time = start['Recording timestamp']
            if next_start_time and start_time < next_start_time:
                continue

            end_time = start_time + WINDOW_SIZE * 1000000
            next_start_time = start_time + STEP * 1000000

            window = raw[(start_time <= raw['Recording timestamp'])
                         & (raw['Recording timestamp'] < end_time)]
            # ここまでOK

            # window = []
            # for index, row in enumerate(raw[start_index:]):
            #     time = datetime.datetime.fromisoformat(row[1])
            #     if time > end_time:
            #         break
            window.append(
                [start_time, label, *list(map(lambda value: float(value), row[2:]))])
            # end_index = start_index + index

            data.extend(window)

            # 末尾到達
            if end_index == len(raw) - 1:
                break

        return data

    raw = pd.read_csv(filename, sep='\t', usecols=USE_COLUMNS)
    data = slide_window(raw, make_label(filename))

    return data


def save_data(filename, data):
    """
    データの保存

    Args:
        filename (string): 保存ファイル名
        data (list): データ
    """

    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['start_id', 'label', 'eyeMoveUp', 'eyeMoveDown',
                        'eyeMoveLeft', 'eyeMoveRight', 'blinkSpeed', 'blinkStrength'])
        writer.writerows(data)


def main():
    files = glob.glob(
        f'{RAW_DIR}/**/*.tsv', recursive=True)

    for filename in files:
        data = preprocess(filename)

        save_file = filename.replace(RAW_DIR, SAVE_DIR)
        save_data(save_file, data)


if __name__ == '__main__':
    main()

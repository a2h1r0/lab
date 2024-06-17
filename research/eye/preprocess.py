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


WINDOW_SIZE = 0.1  # ウィンドウサイズ（秒）
STEP = 0.05  # ステップ幅（秒）


RAW_DIR = f'./data/raw'
SAVE_DIR = f'./data/preprocess/window_{WINDOW_SIZE}'


def preprocess(filename):
    """
    データの前処理

    Args:
        filename (string): ファイル名
    Returns:
        list: 前処理データ
    """

    def slide_window(raw):
        """
        スライディングウィンドウ

        Args:
            raw (list): ローデータ
        Returns:
            list: スライディングウィンドウ後のデータ
        """

        feature = [raw.columns.tolist()]

        next_start_time = None
        for start_index, start in raw.iterrows():
            start_time = start['device_time_stamp']
            if next_start_time and start_time < next_start_time:
                continue

            end_time = start_time + WINDOW_SIZE * 1000000
            next_start_time = start_time + STEP * 1000000

            window = raw[(start_time <= raw['device_time_stamp'])
                         & (raw['device_time_stamp'] < end_time)]
            feature.append(window.iloc[0, :2].values.tolist(
            ) + window.iloc[:, 2:].mean().values.tolist())

            # 末尾到達
            if window.iloc[-1].name == len(raw.index) - 1:
                break

        return feature

    raw = pd.read_csv(filename, header=0)
    feature = slide_window(raw)

    return feature


def save_data(save_file, feature):
    """
    データの保存

    Args:
        save_file (string): 保存ファイル名
        feature (list): 特徴量データ
    """

    if not os.path.exists(os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file))

    with open(save_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(feature)


def main():
    files = glob.glob(
        f'{RAW_DIR}/**/gaze.csv', recursive=True)

    for filename in files:
        feature = preprocess(filename)

        path = filename.split("\\")
        del path[0]

        save_file = f'{SAVE_DIR}/{"_".join(path)}'
        save_data(save_file, feature)


if __name__ == '__main__':
    main()

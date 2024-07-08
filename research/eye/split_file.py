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


WINDOW_SIZE = 10
REMOVE_TIME = 3  # ファイル冒頭削除時間（秒）


RAW_DIR = './data/dropna'
SAVE_DIR = f'./data/split/window_{WINDOW_SIZE}'


def split(filename):
    """
    データの分割

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

        data = []

        next_start_time = None
        for start_index, start in raw.iterrows():
            start_time = start['device_time_stamp'] + REMOVE_TIME * 1000000
            if next_start_time and start_time < next_start_time:
                continue

            end_time = start_time + WINDOW_SIZE * 1000000
            next_start_time = end_time

            window = raw[(start_time <= raw['device_time_stamp'])
                         & (raw['device_time_stamp'] < end_time)]
            data.append(window)

            # 末尾到達
            if window.iloc[-1].name == len(raw.index) - 1:
                break

        return data

    raw = pd.read_csv(filename)
    data = slide_window(raw)

    return data


def save_data(save_dir, data):
    """
    データの保存

    Args:
        save_dir (string): 保存ディレクトリ名
        data (list): データ
    """

    if not os.path.exists(os.path.dirname(save_dir)):
        os.makedirs(os.path.dirname(save_dir))

    for window in data:
        filename = save_dir.replace(
            '.csv', f'_{int(window.iloc[0]["device_time_stamp"])}.csv')
        window.to_csv(filename, index=False)


def main():
    files = glob.glob(
        f'{RAW_DIR}/**/*.csv', recursive=True)

    for filename in files:
        data = split(filename)

        save_dir = filename.replace(RAW_DIR, SAVE_DIR)
        save_data(save_dir, data)


if __name__ == '__main__':
    main()

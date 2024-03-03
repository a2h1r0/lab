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


RAW_DIR = './data/raw'
SAVE_DIR = './data/preprocess/window_30'


WINDOW_SIZE = 30  # ウィンドウサイズ（秒）
STEP = 3  # ステップ幅（秒）


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
        for start_index, start in enumerate(raw):
            start_time = datetime.datetime.fromisoformat(start[1])
            if next_start_time and start_time < next_start_time:
                continue

            start_id = raw[start_index][0]
            end_time = start_time + datetime.timedelta(seconds=WINDOW_SIZE)
            next_start_time = start_time + datetime.timedelta(seconds=STEP)

            window = []
            for index, row in enumerate(raw[start_index:]):
                time = datetime.datetime.fromisoformat(row[1])
                if time > end_time:
                    break
                window.append(
                    [start_id, label, *list(map(lambda value: float(value), row[2:]))])
                end_index = start_index + index

            data.extend(window)

            # 末尾到達
            if end_index == len(raw) - 1:
                break

        return data

    with open(filename) as f:
        reader = csv.reader(f)
        next(reader)
        # データファイルの最終行が空行
        raw = [row[0:8] for row in reader][:-1]

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
    files = glob.glob(f'{RAW_DIR}/**/*.csv', recursive=True)

    for filename in files:
        data = preprocess(filename)

        save_file = filename.replace(RAW_DIR, SAVE_DIR)
        save_data(save_file, data)


if __name__ == '__main__':
    main()

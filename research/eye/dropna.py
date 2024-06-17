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


RAW_DIR = f'./data/raw'
SAVE_DIR = f'./data/dropna'


def dropna(filename):
    """
    NaNの除去

    Args:
        filename (string): ファイル名
    Returns:
        list: 除去済みデータ
    """

    raw = pd.read_csv(filename, header=0)

    return raw.dropna()


def save_data(save_file, data):
    """
    データの保存

    Args:
        save_file (string): 保存ファイル名
        data (list): 保存データ
    """

    if not os.path.exists(os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file))

    data.to_csv(save_file)


def main():
    files = glob.glob(
        f'{RAW_DIR}/**/gaze.csv', recursive=True)

    for filename in files:
        path = filename.split("\\")
        del path[0]

        save_file = f'{SAVE_DIR}/{"_".join(path)}'
        save_data(save_file, dropna(filename))


if __name__ == '__main__':
    main()

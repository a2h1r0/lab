import pandas as pd
import tobii_research as tr
import time
import csv
import os
import datetime


SUBJECT = 'fujii'
COLLECT_TIME = 5


SAVE_DIR = f'../data/{SUBJECT}'


def connect_eyetracker():
    """
    Tobiiの接続

    Returns:
        :tuple:`EyeTracker`: 接続情報

    """

    return tr.find_all_eyetrackers()[0]


def get_gaze_data(gaze_data):
    """
    データの取得

    Args:
        gaze_data (dict): 取得データ
    """

    keys, values = [], []
    for (key, value) in list(gaze_data.items()):
        if isinstance(value, tuple):
            keys.extend([f'{key}_{i + 1}' for i in range(len(value))])
            values.extend(list(value))
        else:
            keys.append(key)
            values.append(value)

    if collection_data == []:
        collection_data.append(keys)
    collection_data.append(values)

    print(f'Timestamp: {gaze_data["device_time_stamp"]}')


def save_data(save_dir):
    """
    データの保存

    Args:
        save_dir (string): 保存ディレクトリ名
    """

    # 保存ディレクトリなんとかする
    print(os.path.exists(os.path.dirname(save_dir)))
    if not os.path.exists(os.path.dirname(save_dir)):
        os.makedirs(os.path.dirname(save_dir))

    now = datetime.datetime.today()
    filename = f'{save_dir}/{now.strftime("%Y%m%d_%H%M%S")}.csv'

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(collection_data)


def main():
    eyetracker = connect_eyetracker()

    eyetracker.subscribe_to(tr.EYETRACKER_GAZE_DATA,
                            get_gaze_data, as_dictionary=True)
    time.sleep(COLLECT_TIME)

    eyetracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, get_gaze_data)

    save_data(SAVE_DIR)


if __name__ == '__main__':
    collection_data = []
    main()

import pandas as pd
import tobii_research as tr
import csv
import os
import datetime
from exam import Exam


QUESTION_NUM = 2

SUBJECT = 'fujii'
SAVE_DIR = f'../data/{SUBJECT}/'


def connect_eyetracker():
    """
    Tobiiの接続

    Returns:
        :tuple:`EyeTracker`: 接続情報
    """

    return tr.find_all_eyetrackers()[0]


def get_collection_data(data):
    """
    データの取得

    Args:
        data (dict): 取得データ
    """

    keys, values = [], []
    for (key, value) in list(data.items()):
        if isinstance(value, tuple):
            keys.extend([f'{key}_{i + 1}' for i in range(len(value))])
            values.extend(list(value))
        else:
            keys.append(key)
            values.append(value)

    if collection_data == []:
        collection_data.append(keys)
    collection_data.append(values)


def save_data(save_dir, gaze_data, answer_data):
    """
    データの保存

    Args:
        save_dir (string): 保存ディレクトリ名
        gaze_data (string): 視線データ
        answer_data (string): 設問回答データ
    """

    if not os.path.exists(os.path.dirname(save_dir)):
        os.makedirs(os.path.dirname(save_dir))

    now = datetime.datetime.today()
    filename = f'{save_dir}{now.strftime("%Y%m%d_%H%M%S")}'

    with open(f'{filename}_gaze.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(gaze_data)

    with open(f'{filename}_answer.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(answer_data)


def main():
    eyetracker = connect_eyetracker()

    print(f'\n\n\n問題は全部で{QUESTION_NUM}問あります．')
    input('\n\n準備ができたらなにかキーを押してください．．．')
    print('\n\n開始します！')
    print('\n\n-----------------------------------')

    eyetracker.subscribe_to(tr.EYETRACKER_GAZE_DATA,
                            get_collection_data, as_dictionary=True)
    answer_data = Exam.calc(QUESTION_NUM)
    eyetracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, get_collection_data)

    print('\n-----------------------------------\n')
    print('\n設問は以上です．ありがとうございました．\n\n')

    save_data(SAVE_DIR, collection_data, answer_data)


if __name__ == '__main__':
    collection_data = []
    main()

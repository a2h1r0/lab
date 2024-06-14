import pandas as pd
import csv
import os
import datetime
from tobii import Tobii
from exam import Exam


QUESTION_NUM = 2

SUBJECT = 'fujii'
SAVE_DIR = f'../data/{SUBJECT}/'


def set_exam_type():
    """
    設問タイプの選択

    Returns:
        string: 設問タイプ
    """

    return int(input('\n\n設問タイプを入力してください > '))


def save_data(save_dir, gaze_data, answer):
    """
    データの保存

    Args:
        save_dir (string): 保存ディレクトリ名
        gaze_data (string): 視線データ
        answer (string): 設問回答データ
    """

    now = datetime.datetime.today()
    save_dir += f'{now.strftime("%Y%m%d_%H%M%S")}/'

    if not os.path.exists(os.path.dirname(save_dir)):
        os.makedirs(os.path.dirname(save_dir))

    with open(f'{save_dir}gaze.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(gaze_data)

    with open(f'{save_dir}answer.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(answer)


def main():
    tobii = Tobii()

    exam_type = set_exam_type()
    if exam_type == 1:
        digit = 1
    elif exam_type == 2:
        digit = 10
    elif exam_type == 3:
        digit = 100
    else:
        digit = 1

    os.system('cls')
    print(f'\n問題は全部で{QUESTION_NUM}問あります．')
    print('\n\n出題を開始します．')
    input('準備ができたらEnterを押してください．．．')

    os.system('cls')
    tobii.subscribe()
    answer = Exam.calc(QUESTION_NUM, digit)
    tobii.unsubscribe()

    os.system('cls')
    print('\n設問は以上です．ありがとうございました．\n\n')

    save_data(SAVE_DIR, tobii.data, answer)


if __name__ == '__main__':
    main()

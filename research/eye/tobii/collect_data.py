import pandas as pd
import csv
import os
import sys
import datetime
from eyetracker import tobii as eyetracker
from exam import Exam
import utils


SUBJECT = 'fujii'
SAVE_DIR = f'../data/raw/{SUBJECT}'


def set_exam_type():
    """
    設問タイプの選択

    Returns:
        string: 設問タイプ
    """

    return utils.input_decimal('\n設問タイプを入力してください > ')


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
    tobii = eyetracker.Tobii()

    print('\n\nキャリブレーションを開始します．．．')
    result = tobii.calibration()

    if result != True:
        print('\nキャリブレーションに失敗しました．\n\n')
        sys.exit(0)
    print('\n完了しました！')

    is_drunk = input('\n\n\nお酒を飲んでいる状態ですか？[Y/n] > ') != 'n'

    exam_type = set_exam_type()
    if exam_type == 1 or exam_type == 2:
        digit = 1 if exam_type == 1 else 10
        question_num = 10

        os.system('cls')
        print(f'\n問題は全部で{question_num}問あります．')
        print('\n\n出題を開始します．')

    elif exam_type == 3:
        digit = 100

    else:
        print('\n中止します．\n\n')
        sys.exit(0)

    input('準備ができたらEnterを押してください．．．')

    os.system('cls')
    tobii.subscribe()

    if exam_type == 1 or exam_type == 2:
        answer = Exam.calc(question_num, digit)

    tobii.unsubscribe()

    os.system('cls')
    if len(tobii.data):
        print('\n設問は以上です．ありがとうございました．\n\n')
        save_data(
            f'{SAVE_DIR}/{"drunk" if is_drunk else "sober"}/exam_type_{exam_type}/', tobii.data, answer)
    else:
        print('\nデータの取得に失敗しました．再取得してください．\n\n')
        sys.exit(0)


if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    main()

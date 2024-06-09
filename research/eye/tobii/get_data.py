import numpy as np
import tobii_research as tr
import time
import csv
import os


COLLECT_TIME = 5
SAVE_DIR = f'../data'


def connect_eyetracker():
    """
    Tobiiの接続

    Returns:
        :tuple:`EyeTracker`: 接続情報

    """

    return tr.find_all_eyetrackers()[0]


def check_not_nan(data):
    return np.count_nonzero(np.isnan(data)) == 0


def get_gaze_data(gaze_data):
    if check_not_nan(gaze_data['left_gaze_point_on_display_area']) and check_not_nan(gaze_data['right_gaze_point_on_display_area']):
        collection_data.append(
            gaze_data['left_gaze_point_on_display_area'] + gaze_data['right_gaze_point_on_display_area'])
        print(
            f'Left eye: ({gaze_data["left_gaze_point_on_display_area"]}) \t Right eye: ({gaze_data["right_gaze_point_on_display_area"]})')


def save_data(save_dir):
    """
    データの保存

    Args:
        save_dir (string): 保存ディレクトリ名
    """

    if not os.path.exists(os.path.dirname(save_dir)):
        os.makedirs(os.path.dirname(save_dir))

    filename = f'{save_dir}/test.csv'

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['left_gaze_point_on_display_area_x', 'left_gaze_point_on_display_area_y',
                        'right_gaze_point_on_display_area_x', 'right_gaze_point_on_display_area_y'])
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

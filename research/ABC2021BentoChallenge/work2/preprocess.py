import numpy as np

# ウィンドウサイズとオーバーラップサイズを個数指定(1個あたり10ms)
WINDOW_SIZE = 40  # 個数指定 (50ms ならば 5)
OVERLAP = -5  # 個数指定 (-30ms ならば -3)


def make_feature(raw_data, use_markers):
    """
    特徴量の作成
    平均値 / 分散値 / 最大値 / 最小値
    RMS / IQR / ZCR

    Args:
        raw_data (array): 生データ
        use_markers (array): 使用する部位
    Returns:
        array: 特徴量データ
    """

    BODY_PARTS = ['front_head', 'top_head', 'rear_head', 'right_shoulder', 'right_offset',
                  'right_elbow', 'right_wrist', 'left_shoulder', 'left_elbow', 'left_wrist',
                  'right_asis', 'left_asis', 'v_sacral']
    LABELS = ['X', 'Y', 'Z']

    number_list = []
    for i in range(len(use_markers)):
        for j in range(len(BODY_PARTS)):
            if(BODY_PARTS[j] == use_markers[i]):
                number_list.append(int(j))
                break

    feature_data = []
    for i in range(len(number_list)):
        feature_data.append([])

        counter = 0
        while(True):
            if(counter + WINDOW_SIZE >= len(raw_data)):
                break

            # スライディングウィンドウの切り出し
            split_data = raw_data[counter:counter + WINDOW_SIZE]

            # 使用する部位・軸ごとに各特徴量を計算(7種類×3軸)
            feature_data[i].append([])

            for j in range(len(LABELS)):
                value = []
                for k in range(len(split_data)):
                    value.append(float(split_data[k][3 * number_list[i] + j + 1]))

                # 平均値
                feature_data[i][-1].append(np.average(value))

                # 分散値
                feature_data[i][-1].append(np.var(value))

                # 最大値
                feature_data[i][-1].append(np.max(value))

                # 最小値
                feature_data[i][-1].append(np.min(value))

                # RMS
                feature_data[i][-1].append(np.sqrt(np.sum(np.square(value)) / WINDOW_SIZE))

                # IQR
                percent_75, percent_25 = np.percentile(value, [75, 25])
                feature_data[i][-1].append(percent_75 - percent_25)

                # ZCR
                zero_cross = 0
                for k in range(len(value) - 1):
                    if(value[k] * value[k + 1] < 0):
                        zero_cross += 1
                feature_data[i][-1].append(zero_cross / len(value))

            counter += WINDOW_SIZE + OVERLAP

    return feature_data


def make_raw(raw_data, use_markers):
    """
    生データの作成

    Args:
        raw_data (array): 生データ
        use_markers (array): 使用する部位
    Returns:
        array: 整形済み生データ
    """

    BODY_PARTS = ['front_head', 'top_head', 'rear_head', 'right_shoulder', 'right_offset',
                  'right_elbow', 'right_wrist', 'left_shoulder', 'left_elbow', 'left_wrist',
                  'right_asis', 'left_asis', 'v_sacral']
    LABELS = ['X', 'Y', 'Z']

    number_list = []
    for i in range(len(use_markers)):
        for j in range(len(BODY_PARTS)):
            if(BODY_PARTS[j] == use_markers[i]):
                number_list.append(int(j))
                break

    feature_data = []
    for i in range(len(number_list)):
        feature_data.append([])

        for j in range(len(LABELS)):
            value = []
            for k in range(len(raw_data)):
                value.append(float(raw_data[k][3 * number_list[i] + j + 1]))

            feature_data[i].append(value)

        feature_data[i] = np.array(feature_data[i]).T

    return feature_data


if __name__ == '__main__':
    import csv
    import os
    os.chdir(os.path.dirname(__file__))

    FILE = '../dataset/train/acceleration/1_13/subject_1_activity_1_repeat_1.csv'
    USE_MARKERS = ['right_shoulder', 'left_wrist']

    with open(FILE) as f:
        reader = csv.reader(f)
        next(reader)
        raw_data = [row for row in reader]
        # feature_data = make_feature(raw_data, USE_MARKERS)
        feature_data = make_raw(raw_data, USE_MARKERS)

    print(feature_data)

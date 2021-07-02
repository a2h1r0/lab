def make_feature(raw_data, use_markers):
    """
    特徴量の作成

    Args:
        raw_data (array): 生データ
        use_markers (array): 使用する部位
    Returns:
        array: 特徴量データ
    """

    feature_data = raw_data

    return feature_data


if __name__ == '__main__':
    import csv
    import os
    os.chdir(os.path.dirname(__file__))

    FILENAME = './dataset/train/acceleration/1_13/subject_1_activity_1_repeat_1.csv'
    USE_MARKERS = ['right_shoulder', 'left_wrist']

    with open(FILENAME) as f:
        reader = csv.reader(f)
        next(reader)
        raw_data = [row for row in reader]
        feature_data = make_feature(raw_data, USE_MARKERS)

    print(feature_data)

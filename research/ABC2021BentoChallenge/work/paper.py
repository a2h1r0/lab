from preprocess import make_feature
import matplotlib.pyplot as plt
import csv
import os
os.chdir(os.path.dirname(__file__))


FILENAME = '../dataset/train/autocorrelation/subject_1_activity_1_001.csv'
SAVENAME = 'wave_feature_1'

USE_MARKERS = ['right_shoulder', 'right_elbow', 'right_wrist',
               'left_shoulder', 'left_elbow', 'left_wrist']


def main():
    with open(FILENAME) as f:
        reader = csv.reader(f)
        next(reader)
        raw_data = [row for row in reader if '' not in row]
        # feature_data = make_raw(raw_data, USE_MARKERS)
        feature_data = make_feature(raw_data, USE_MARKERS)

    figures_dir = '../figures/'
    plt.figure(figsize=(9, 6))
    # for marker, data in zip(USE_MARKERS, feature_data):
    #     plt.plot(range(len(data)), data, label=marker)
    plt.plot(range(len(feature_data[0])), feature_data[0])
    # plt.xlabel('Epoch', fontsize=26)
    # plt.ylabel('Loss', fontsize=26)
    # plt.legend(fontsize=26, loc='upper right')
    # plt.tick_params(labelsize=26)
    plt.xticks(color='None')
    plt.yticks(color='None')
    plt.tick_params(length=0)
    plt.savefig(figures_dir + SAVENAME + '.svg', bbox_inches='tight', pad_inches=0)
    plt.savefig(figures_dir + SAVENAME + '.eps', bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == '__main__':
    main()

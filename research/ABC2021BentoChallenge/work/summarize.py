import numpy as np
import pandas as pd
from natsort import natsorted
import glob
import os
os.chdir(os.path.dirname(__file__))


DATA_DIR = '../data/20210726_202222/'


def main():
    data = []
    dirs = natsorted(os.listdir(DATA_DIR))
    for dir_name in dirs:
        data.append([])
        files = natsorted(glob.glob(DATA_DIR + dir_name + '/report_*.csv'))
        for index, filename in enumerate(files):
            report_df = pd.read_csv(filename, index_col=0)
            score = report_df.loc['f1-score', 'accuracy']
            data[-1].append([filename, score])
    data = np.array(data).T
    filename_all = data[0]
    score_all = data[1]
    for filenames, scores in zip(filename_all, score_all):
        print('\n----- ' + filenames[0].split('\\')[1] + ' -----')
        # for filename, score in zip(filenames, scores):
        #     print(filename.split('\\')[0].split('/')[-1] + ': ' + score)
        max_index = np.argmax(scores)
        print('max score: ' + scores[max_index])
        print('max filename: ' + filenames[max_index])


if __name__ == '__main__':
    main()

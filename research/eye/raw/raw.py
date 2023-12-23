import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import os
os.chdir(os.path.dirname(__file__))


FILENAME = '20231223-153208_C152633DCDAE_currentData.csv'
DATA_DIR = '../data/'


def main():
    df = pd.read_csv(DATA_DIR + FILENAME, index_col='id')[:1000]
    df['date'] = df['date'].apply(lambda date: datetime.fromisoformat(date))

    fig = plt.figure(figsize=(16, 9))

    eye_move = fig.add_subplot(2, 1, 1)
    eye_move.set_title('eye move')
    eye_move.set_xlabel('index')
    eye_move.set_ylabel('range')
    eye_move.set_xlim(df['date'][0], df['date'][len(df) - 1])
    eye_move.set_ylim(-8, 8)

    eye_move.plot(df['date'], df['eyeMoveUp'] -
                  df['eyeMoveDown'], 'red', label='up down')
    eye_move.plot(df['date'], df['eyeMoveLeft'] -
                  df['eyeMoveRight'], 'blue', label='left right')
    eye_move.legend()

    blink = fig.add_subplot(2, 1, 2)
    blink.set_title('blink')
    blink.set_xlabel('index')
    blink.set_ylabel('power')
    blink.set_xlim(df['date'][0], df['date'][len(df) - 1])
    blink.set_ylim(0, 300)

    blink.plot(df['date'], df['blinkSpeed'], 'red', label='blink speed')
    blink.plot(df['date'], df['blinkStrength'], 'blue', label='blink strength')
    blink.legend()

    plt.subplots_adjust(top=0.85, hspace=0.5)
    # plt.savefig('../figures/power.png', bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == '__main__':
    main()

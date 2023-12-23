import pandas as pd
import matplotlib.pyplot as plt
import os
os.chdir(os.path.dirname(__file__))


FILENAME = '20231223-153208_C152633DCDAE_currentData.csv'
DATA_DIR = '../data/'


def main():
    df = pd.read_csv(DATA_DIR + FILENAME, index_col='id')[:1000]

    fig = plt.figure(figsize=(16, 9))

    eye_move = fig.add_subplot(2, 1, 1)
    eye_move.set_title('eye move')
    eye_move.set_xlabel('index')
    eye_move.set_ylabel('range')
    eye_move.set_xlim(0, len(df.index))
    eye_move.set_ylim(-8, 8)

    eye_move.plot(range(len(df.index)), df['eyeMoveUp'] -
                  df['eyeMoveDown'], 'red', label='up down')
    eye_move.plot(range(len(df.index)),
                  df['eyeMoveLeft'] - df['eyeMoveRight'], 'blue', label='left right')
    eye_move.legend()

    blink = fig.add_subplot(2, 1, 2)
    blink.set_title('blink')
    blink.set_xlabel('index')
    blink.set_ylabel('power')
    blink.set_xlim(0, len(df.index))
    blink.set_ylim(0, 300)

    blink.plot(range(len(df.index)),
               df['blink speed'], 'red', label='blinkSpeed')
    blink.plot(range(len(df.index)),
               df['blink strength'], 'blue', label='blinkStrength')
    blink.legend()

    plt.subplots_adjust(top=0.85, hspace=0.5)
    # plt.savefig('../figures/power.png', bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == '__main__':
    main()

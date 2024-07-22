import glob
import shutil
import os
os.chdir(os.path.dirname(__file__))


RAW_DIR = './data/raw'
SAVE_DIR = './data/raw_exam'


def copy_file(filename, save_file):
    """
    ファイルのコピー

    Args:
        filename (string): コピー元ファイル名
        save_file (string): コピー先ファイル名
    """

    if not os.path.exists(os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file))

    shutil.copy(filename, save_file)


def main():
    files = glob.glob(
        f'{RAW_DIR}/**/gaze.csv', recursive=True)

    for filename in files:
        path = filename.split("\\")
        del path[0]

        save_file = f'{SAVE_DIR}/{path[2]}/{path[0]}/{path[1]}_{path[3]}_{path[4]}'
        copy_file(filename, save_file)


if __name__ == '__main__':
    main()

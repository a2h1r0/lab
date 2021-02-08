import matplotlib.pyplot as plt
import csv
import glob
import os
os.chdir(os.path.dirname(__file__))


def archive_csv(filename, step, delete_source=False):
    """CSVファイルの圧縮
    GitHubに保存するために100MB以下に分割する．

    Args:
        filename (string): 圧縮するファイル
        step (int): 1ファイルに格納するエポック数
        delete_source (boolean): 元ファイルの削除
    """

    flag = True
    file_count = 0
    with open('./data/' + filename + '.csv') as f:
        reader = csv.reader(f)
        header = next(reader)

        for row in reader:
            if int(row[0]) % step == 1 and flag:
                flag = False
                file_count += step
                # すでに開いているファイルがあればクローズ
                if 'new_file' in locals():
                    new_file.close()
                new_file = open('./data/' + filename +
                                '_' + str(file_count) + '.csv', 'a', newline='')
                new_writer = csv.writer(new_file, delimiter=',')
                new_writer.writerow(header)

            if int(row[0]) % step == 0:
                flag = True

            new_writer.writerow(row)

    # 元ファイルの削除
    if delete_source:
        os.remove('./data/' + filename + '.csv')


def plot_csv(time, max_epoch, step, savefig=True):
    """CSVファイルの脈波の描画

    Args:
        time (string): 描画するファイルの日時
        max_epoch (int): 描画する最大エポック数
        step (int): 何エポックごとに描画するか
        savefig (boolean): 図表の保存
    """

    # データの読み出し
    t = [[] for i in range(0, max_epoch, step)]
    y_generated = [[] for i in range(0, max_epoch, step)]
    files = glob.glob('./data/' + time + '_generated_*.csv')
    for index, data in enumerate(files):
        with open(data) as f:
            reader = csv.reader(f)
            next(reader)

            for row in reader:
                if int(row[0]) > max_epoch:
                    break

                if int(row[0]) % step != 0:
                    continue
                elif int(row[0]) % step == 0:
                    t[index].append(float(row[1]) / 1000)
                    y_generated[index].append(int(row[2]))

    y_raw = [[] for i in range(0, max_epoch, step)]
    files = glob.glob('./data/' + time + '_raw_*.csv')
    for index, data in enumerate(files):
        with open(data) as f:
            reader = csv.reader(f)
            next(reader)

            for row in reader:
                if int(row[0]) > max_epoch:
                    break

                if int(row[0]) % step != 0:
                    continue
                elif int(row[0]) % step == 0:
                    y_raw[index].append(int(row[1]))

    # データの描画
    for index in range(len(t)):
        epoch = str((index + 1) * step)

        plt.figure(figsize=(16, 9))
        plt.plot(t[index], y_generated[index], 'red', label='Generated')
        plt.plot(t[index], y_raw[index], 'blue', label='Raw')
        plt.xlabel('Time [s]', fontsize=18)
        plt.ylabel('Pulse sensor value', fontsize=18)
        plt.title('Epoch: ' + epoch)
        plt.tick_params(labelsize=18)
        plt.legend(fontsize=18, loc='upper right')
        if savefig:
            plt.savefig('../figure/' + time + '_' + str(len(t[index])) + '_' + epoch + 'epoch.png',
                        bbox_inches='tight', pad_inches=0)

    plt.show()


if __name__ == '__main__':
    archive_csv('20210208_004429_generated', step=1000, delete_source=True)
    archive_csv('20210208_004429_raw', step=1000, delete_source=True)

    plot_csv('20210208_004429', max_epoch=5000, step=1000, savefig=False)

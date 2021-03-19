import matplotlib.pyplot as plt
import csv
from natsort import natsorted
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
    with open(filename) as f:
        reader = csv.reader(f)
        header = next(reader)

        for row in reader:
            if int(row[0]) % step == 1 and flag:
                flag = False
                file_count += step
                # すでに開いているファイルがあればクローズ
                if 'new_file' in locals():
                    new_file.close()
                new_file = open('.' + filename.split('.')[1] + '_' +
                                str(file_count) + '.csv', 'a', newline='')
                new_writer = csv.writer(new_file, delimiter=',')
                new_writer.writerow(header)

            if int(row[0]) % step == 0:
                flag = True

            new_writer.writerow(row)

    # 元ファイルの削除
    if delete_source:
        os.remove(filename)


def plot_pulse_csv(file_dir, max_epoch, step, savefig=True):
    """CSVファイルの脈波の描画

    Args:
        file_dir (string): 描画するファイルのディレクトリ
        max_epoch (int): 描画する最大エポック数
        step (int): 何エポックごとに描画するか（< 1ファイルのエポック数）
        savefig (boolean): 図表の保存
    """

    # データの読み出し
    t = [[] for i in range(0, max_epoch, step)]
    y_generated = [[] for i in range(0, max_epoch, step)]
    files = natsorted(glob.glob(file_dir + '/generated_*.csv'))
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
    files = natsorted(glob.glob(file_dir + '/raw_*.csv'))
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
            plt.savefig('../../figure/' + file_dir.split('/')[-1] + '_' + str(len(t[index])) + '_' + epoch + 'epoch.png',
                        bbox_inches='tight', pad_inches=0)
    plt.show()


def plot_colors_csv(file_dir, max_epoch, step, savefig=True):
    """CSVファイルの色データの描画

    Args:
        file_dir (string): 描画するファイルのディレクトリ
        max_epoch (int): 描画する最大エポック数
        step (int): 何エポックごとに描画するか（< 1ファイルのエポック数）
        savefig (boolean): 図表の保存
    """

    # データの読み出し
    y_real = [[] for i in range(0, max_epoch, step)]
    y_fake = [[] for i in range(0, max_epoch, step)]
    files = natsorted(glob.glob(file_dir + '/colors_*.csv'))
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
                    y_real[index].append(int(row[1]))
                    y_fake[index].append(int(row[2]))

    # データの描画
    for index in range(len(y_real)):
        epoch = str((index + 1) * step)

        plt.figure(figsize=(16, 9))
        plt.plot(range(len(y_real[index])),
                 y_real[index], 'blue', label='Real')
        plt.plot(range(len(y_fake[index])),
                 y_fake[index], 'red', label='Fake')
        plt.xlabel('Time [s]', fontsize=18)
        plt.ylabel('Pulse sensor value', fontsize=18)
        plt.title('Epoch: ' + epoch)
        plt.tick_params(labelsize=18)
        plt.legend(fontsize=18, loc='upper right')
        if savefig:
            plt.savefig('../figure/' + file_dir.split('/')[-1] + '_' + str(len(t[index])) + '_' + epoch + 'epoch.png',
                        bbox_inches='tight', pad_inches=0)
    plt.show()


def plot_loss_csv(file_dir, save_figname=False):
    """CSVファイルのLossの描画

    Args:
        file_dir (string): 描画するファイルのディレクトリ
        save_figname (string): 図表の保存ファイル名
    """

    epoch = []
    D_loss = []
    G_loss = []
    with open(file_dir + '/loss.csv') as f:
        reader = csv.reader(f)

        # ヘッダーのスキップ
        next(reader)

        for row in reader:
            # データの追加
            epoch.append(float(row[0]))
            D_loss.append(float(row[1]))
            G_loss.append(float(row[2]))

    plt.figure(figsize=(16, 9))
    plt.plot(epoch, D_loss, 'red', label="Discriminator")
    plt.plot(epoch, G_loss, 'blue', label="Generator")
    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("Loss", fontsize=18)
    plt.tick_params(labelsize=18)
    plt.legend(fontsize=18, loc='upper right')
    if save_figname is not False:
        plt.savefig('../figure/' + save_figname,
                    bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == '__main__':
    archive_csv('./data/20210310_173919/generated.csv',
                step=500, delete_source=True)
    archive_csv('./data/20210310_173919/raw.csv',
                step=500, delete_source=True)

    plot_pulse_csv('./data/20210310_173919',
                   max_epoch=5000, step=500, savefig=False)

    # plot_loss_csv('./data/20210213_153255')
    # plot_loss_csv('./data/20210209_010240', '256_generated_1400epoch_loss.png')

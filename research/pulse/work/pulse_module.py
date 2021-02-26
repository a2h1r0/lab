import matplotlib.pyplot as plt
import csv
import os
os.chdir(os.path.dirname(__file__))


def plot_colors_csv(file_dir, max_epoch, step, savefig=True):
    """CSVファイルの色データの描画

    Args:
        file_dir (string): 描画するファイルのディレクトリ
        max_epoch (int): 描画する最大エポック数
        step (int): 何エポックごとに描画するか
        savefig (boolean): 図表の保存
    """

    # データの読み出し
    y_real = [[] for i in range(0, max_epoch, step)]
    y_fake = [[] for i in range(0, max_epoch, step)]
    with open(file_dir + '/colors.csv') as f:
        reader = csv.reader(f)
        next(reader)

        for row in reader:
            if int(row[0]) > max_epoch:
                break

            if int(row[0]) % step != 0:
                continue
            elif int(row[0]) % step == 0:
                y_real[int(row[0]) // step - 1].append(int(row[1]))
                y_fake[int(row[0]) // step - 1].append(int(row[2]))

    # データの描画
    for index in range(len(y_real)):
        epoch = str((index + 1) * step)

        plt.figure(figsize=(16, 9))
        plt.plot(list(range(len(y_real[index]))),
                 y_real[index], 'blue', label='Real')
        plt.plot(list(range(len(y_fake[index]))),
                 y_fake[index], 'red', label='Fake')
        plt.xlabel('Time [s]', fontsize=18)
        plt.ylabel('Gray Scale', fontsize=18)
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
    plot_colors_csv('./data/20210226_231141',
                    max_epoch=1000, step=100, savefig=False)

    plot_loss_csv('./data/20210226_231141')
    # plot_loss_csv('./data/20210226_231141', '256_generated_1400epoch_loss.png')

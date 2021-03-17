import matplotlib.pyplot as plt
import csv
import os
os.chdir(os.path.dirname(__file__))


def plot_data_csv(file_dir, max_epoch, step, savefig=True):
    """CSVファイルのデータの描画

    Args:
        file_dir (string): 描画するファイルのディレクトリ
        max_epoch (int): 描画する最大エポック数
        step (int): 何エポックごとに描画するか
        savefig (boolean): 図表の保存
    """

    # データの読み出し
    y_real = [[] for i in range(0, max_epoch, step)]
    y_fake = [[] for i in range(0, max_epoch, step)]
    y_pulse = [[] for i in range(0, max_epoch, step)]
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
                y_pulse[int(row[0]) // step - 1].append(int(row[3]))

    # データの描画
    for index in range(len(y_real)):
        epoch = str((index + 1) * step)

        fig, ax1 = plt.subplots(figsize=(16, 9))
        ax1.plot(list(range(len(y_real[index]))),
                 y_real[index], 'blue', label='Real')
        ax1.plot(list(range(len(y_fake[index]))),
                 y_fake[index], 'red', label='Fake')
        ax2 = ax1.twinx()
        ax2.plot(list(range(len(y_pulse[index]))),
                 y_pulse[index], 'green', label='Pulse')
        ax1.set_xlabel('Time [s]', fontsize=18)
        ax1.set_ylabel('Gray Scale', fontsize=18)
        ax2.set_ylabel('Pulse Value', fontsize=18)
        plt.title('Epoch: ' + epoch)
        plt.tick_params(labelsize=18)
        handler1, label1 = ax1.get_legend_handles_labels()
        handler2, label2 = ax2.get_legend_handles_labels()
        ax1.legend(handler1 + handler2, label1 + label2,
                   fontsize=18, loc='upper right')
        if savefig:
            plt.savefig('../figure/' + file_dir.split('/')[-1] + '_' + str(len(y_pulse[index])) + '_' + epoch + 'epoch.png',
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
    plot_data_csv('./data/20210317_165322',
                  max_epoch=100000, step=10000, savefig=True)

    # plot_loss_csv('./data/20210226_231141')
    # plot_loss_csv('./data/20210226_231141', '256_generated_1400epoch_loss.png')

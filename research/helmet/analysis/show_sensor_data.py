import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
os.chdir(os.path.dirname(__file__))

tester = ['ooyama', 'okamoto']  # **被験者**
colors = ['red', 'blue']
linestyles = ['solid', 'dashed']

fig, ax = plt.subplots(32, 1, figsize=(9, 16), sharex='all')
plt.subplots_adjust(hspace=1.0)

## データの読み込み ##
for name, color, linestyle in zip(tester, colors, linestyles):   # 被験者1人ずつデータを読み込む
    filename = './dataset/' + name + '.csv'
    data = pd.read_csv(filename, usecols=['in0', 'in1', 'in2', 'in3', 'in4', 'in5', 'in6', 'in7', 'in8', 'in9', 'in10',
                                          'in11', 'in12', 'in13', 'in14', 'in15', 'in16', 'in17', 'in18', 'in19', 'inA',
                                          'inB', 'inC', 'inD', 'inE', 'inF', 'inあ', 'inい', 'inう', 'inア', 'inイ', 'inウ', 'Number'], encoding='Shift-JIS')
    data.fillna(0, inplace=True)    # 区切り番号以外'0'で埋める
    # センサ順に並び替え
    data = data[['in0', 'in1', 'in2', 'in3', 'in4', 'in5', 'in6', 'in7', 'in8', 'in9', 'in10',
                 'in11', 'in12', 'in13', 'in14', 'in15', 'in16', 'in17', 'in18', 'in19', 'inA',
                 'inB', 'inC', 'inD', 'inE', 'inF', 'inあ', 'inい', 'inう', 'inア', 'inイ', 'inウ', 'Number']]

    values = []
    for row in data.itertuples(name=None):  # 1行ずつ読み出し
        row = list(row)

        # 1回目だけ使用
        if row[-1] == 2:
            break

        values.append(row)

    # 転置
    values = np.array(values).T.tolist()
    del values[0]
    del values[-1]
    for i in range(len(values)):
        ax[i].plot(range(len(values[i])), values[i],
                   color, linestyle=linestyle)
        ax[i].set_xlim(0, len(values[i]))
        ax[i].set_ylim(0, 5)
        ax[i].set_ylabel('#' + str(i), fontsize=12)
        ax[i].tick_params(labelsize=12)


ax[31].set_xlabel('Sumple Number', fontsize=18)
ax[31].plot([], [], color=colors[0],
            linestyle=linestyles[0], label='Subject A')
ax[31].plot([], [], color=colors[1],
            linestyle=linestyles[1], label='Subject B')
ax[31].legend(bbox_to_anchor=(0, -1.5), loc='upper left',
              borderaxespad=0, fontsize=18)

plt.savefig('./sensor_data.svg', bbox_inches='tight', pad_inches=0)
plt.show()

import numpy as np
import pandas as pd
import csv




label_num = 10


## データの読み込み ##
filename = 'left_hip.tsv'
left_hip_df = pd.read_table(filename, usecols=['Filename', 'Groundtruth', 'EPOCH999'], index_col=0, encoding='Shift-JIS')

filename = 'right_arm.tsv'
right_arm_df = pd.read_table(filename, usecols=['Filename', 'Groundtruth', 'EPOCH999'], index_col=0, encoding='Shift-JIS')

filename = 'left_wrist.tsv'
left_wrist_df = pd.read_table(filename, usecols=['Filename', 'Groundtruth', 'EPOCH999'], index_col=0, encoding='Shift-JIS')

filename = 'right_wrist.tsv'
right_wrist_df = pd.read_table(filename, usecols=['Filename', 'Groundtruth', 'EPOCH999'], index_col=0, encoding='Shift-JIS')




# ファイルの存在する部位だけ読み込み
# F値の比率計算
# フラグに掛け算，加算してファイルに書き出す
with open('out_flags.csv', 'w', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['ファイル', '正解', '動作1', '動作2', '動作3', '動作4', '動作5', '動作6', '動作7', '動作8', '動作9', '動作10'])


with open('labels.csv', 'r') as f:
    reader = csv.reader(f)
    all_files = [row[0] for row in reader]
    
    
for file in all_files:
    filename = file
    
    parts = ['left_hip', 'right_arm', 'left_wrist', 'right_wrist']
    preds = [[] for part in parts]

    f1_scores = np.zeros(4)
    f1_scores[0] = left_hip_df.at['F1 score', 'EPOCH999']
    f1_scores[1] = right_arm_df.at['F1 score', 'EPOCH999']
    f1_scores[2] = left_wrist_df.at['F1 score', 'EPOCH999']
    f1_scores[3] = right_wrist_df.at['F1 score', 'EPOCH999']
    
    
    isfile = left_hip_df.index.str.endswith(filename)
    if True in isfile and left_hip_df.at[filename, 'EPOCH999'] != '[]':
        groundtruth = list(map(int, left_hip_df.at[filename, 'Groundtruth'].strip('['']').split(', ')))
        preds[0] = list(map(int, left_hip_df.at[filename, 'EPOCH999'].strip('['']').split(', ')))
    else:
        f1_scores[0] = 0
    
    isfile = right_arm_df.index.str.endswith(filename)
    if True in isfile and right_arm_df.at[filename, 'EPOCH999'] != '[]':
        groundtruth = list(map(int, right_arm_df.at[filename, 'Groundtruth'].strip('['']').split(', ')))
        preds[1] = list(map(int, right_arm_df.at[filename, 'EPOCH999'].strip('['']').split(', ')))
    else:
        f1_scores[1] = 0
    
    isfile = left_wrist_df.index.str.endswith(filename)
    if True in isfile and left_wrist_df.at[filename, 'EPOCH999'] != '[]':
        groundtruth = list(map(int, left_wrist_df.at[filename, 'Groundtruth'].strip('['']').split(', ')))
        preds[2] = list(map(int, left_wrist_df.at[filename, 'EPOCH999'].strip('['']').split(', ')))
    else:
        f1_scores[2] = 0
    
    isfile = right_wrist_df.index.str.endswith(filename)
    if True in isfile and right_wrist_df.at[filename, 'EPOCH999'] != '[]':
        groundtruth = list(map(int, right_wrist_df.at[filename, 'Groundtruth'].strip('['']').split(', ')))
        preds[3] = list(map(int, right_wrist_df.at[filename, 'EPOCH999'].strip('['']').split(', ')))
    else:
        f1_scores[3] = 0
    
    
    
    flags = [0 for i in range(label_num)]
    f1_ratio = f1_scores / max(f1_scores)
    
    for part, pred in enumerate(preds):
        if pred == []:
            continue
    
        else:
            for index in pred:
                flags[index] += f1_ratio[part]
    
    row = flags
    row.insert(0, groundtruth)
    row.insert(0, filename)
    with open('out_flags.csv', 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)
    




# フラグのデータを読み出す
filename = 'out_flags.csv'
flags_df = pd.read_csv(filename, index_col=0, encoding='Shift-JIS')


# 最大になる閾値の計算
thresholds = np.arange(0.5, 3.1, 0.1)
max_threshold = []
for threshold in thresholds:
    num = 0
    for flags in flags_df.itertuples():
        file = list(flags)[0]
        groundtruth = list(map(int, list(flags)[1].strip('['']').split(', ')))
        flag = np.array(flags[2:])
    
        pred = list(*np.where((flag != 0) & (flag > threshold)))
    
        if groundtruth == pred:
            num += 1
    max_threshold.append(num)
print('\n完全正解：', end='')
print(max(max_threshold))
threshold = thresholds[max_threshold.index(max(max_threshold))]


# 閾値最大で予測データのフラグをラベルに戻す
# 結果を書き出す
with open('out_preds.csv', 'w', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['ファイル', '正解', 'pred'])

for flags in flags_df.itertuples():
    file = list(flags)[0]
    groundtruth = list(map(int, list(flags)[1].strip('['']').split(', ')))
    flag = np.array(flags[2:])

    pred = list(*np.where((flag != 0) & (flag > threshold)))

    row = file + ':[' + ', '.join(map(str, groundtruth)) + ']:[' + ', '.join(map(str, pred)) + ']'
    row = row.split(':')

    with open('out_preds.csv', 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

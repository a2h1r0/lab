import numpy as np
import pandas as pd
import csv




###*** ラベル化されたデータ(1,0)に重みを掛ける手法 ***###
print("\n--手法1--\n")

##** 行動 **##
label_num = 10

## データの読み込み ##
filename = 'Activity_predictionleft_hip.tsv'
left_hip_df = pd.read_table(filename, usecols=['Filename', 'Groundtruth', 'EPOCH1000'], index_col=0, encoding='Shift-JIS')

filename = 'Activity_predictionright_arm.tsv'
right_arm_df = pd.read_table(filename, usecols=['Filename', 'Groundtruth', 'EPOCH1000'], index_col=0, encoding='Shift-JIS')

filename = 'Activity_predictionleft_wrist.tsv'
left_wrist_df = pd.read_table(filename, usecols=['Filename', 'Groundtruth', 'EPOCH1000'], index_col=0, encoding='Shift-JIS')

filename = 'Activity_predictionright_wrist.tsv'
right_wrist_df = pd.read_table(filename, usecols=['Filename', 'Groundtruth', 'EPOCH1000'], index_col=0, encoding='Shift-JIS')




# ファイルの存在する部位だけ読み込み
# F値の比率計算
# フラグに掛け算，加算してファイルに書き出す
with open('out_flags.csv', 'w', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['ファイル', '正解', '動作1', '動作2', '動作3', '動作4', '動作5', '動作6', '動作7', '動作8', '動作9', '動作10'])


with open('labels.csv', 'r') as f:
    reader = csv.reader(f)
    all_files_temp = [row[0] for row in reader]
all_files = [s for s in all_files_temp if s.startswith('subject2')]
    
    
for file in all_files:
    filename = file
    
    parts = ['left_hip', 'right_arm', 'left_wrist', 'right_wrist']
    preds = [[] for part in parts]

    f1_scores = np.zeros(4)
    f1_scores[0] = left_hip_df.at['Test Accuracy', 'EPOCH1000']
    f1_scores[1] = right_arm_df.at['Test Accuracy', 'EPOCH1000']
    f1_scores[2] = left_wrist_df.at['Test Accuracy', 'EPOCH1000']
    f1_scores[3] = right_wrist_df.at['Test Accuracy', 'EPOCH1000']
    
    
    isfile = left_hip_df.index.str.endswith(filename)
    if True in isfile and left_hip_df.at[filename, 'EPOCH1000'] != '[]':
        groundtruth = list(map(int, left_hip_df.at[filename, 'Groundtruth'].strip('['']').split(', ')))
        preds[0] = list(map(int, left_hip_df.at[filename, 'EPOCH1000'].strip('['']').split(', ')))
    else:
        f1_scores[0] = 0
    
    isfile = right_arm_df.index.str.endswith(filename)
    if True in isfile and right_arm_df.at[filename, 'EPOCH1000'] != '[]':
        groundtruth = list(map(int, right_arm_df.at[filename, 'Groundtruth'].strip('['']').split(', ')))
        preds[1] = list(map(int, right_arm_df.at[filename, 'EPOCH1000'].strip('['']').split(', ')))
    else:
        f1_scores[1] = 0
    
    isfile = left_wrist_df.index.str.endswith(filename)
    if True in isfile and left_wrist_df.at[filename, 'EPOCH1000'] != '[]':
        groundtruth = list(map(int, left_wrist_df.at[filename, 'Groundtruth'].strip('['']').split(', ')))
        preds[2] = list(map(int, left_wrist_df.at[filename, 'EPOCH1000'].strip('['']').split(', ')))
    else:
        f1_scores[2] = 0
    
    isfile = right_wrist_df.index.str.endswith(filename)
    if True in isfile and right_wrist_df.at[filename, 'EPOCH1000'] != '[]':
        groundtruth = list(map(int, right_wrist_df.at[filename, 'Groundtruth'].strip('['']').split(', ')))
        preds[3] = list(map(int, right_wrist_df.at[filename, 'EPOCH1000'].strip('['']').split(', ')))
    else:
        f1_scores[3] = 0
    
    if all([i == 0 for i in f1_scores]):
        continue
    
    
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
thresholds = np.arange(0.5, 3.5, 0.01)
correct = []
for threshold in thresholds:
    num = 0
    for flags in flags_df.itertuples():
        file = flags[0]
        groundtruth = list(map(int, list(flags)[1].strip('['']').split(', ')))
        flag = np.array(flags[2:])
    
        pred = list(*np.where((flag != 0) & (flag > threshold)))
        if pred == []:
            sub = 0.0
            while pred == []:
                pred = list(*np.where((flag != 0) & (flag > threshold-sub)))
                sub += 0.1

        if groundtruth == pred:
            num += 1
    correct.append(num)
print('\n完全正解：', end='')
print(max(correct))
threshold = thresholds[correct.index(max(correct))]
print(threshold)


# 閾値最大で予測データのフラグをラベルに戻す
# 結果を書き出す
with open('out_preds.csv', 'w', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['ファイル', '正解', 'pred'])

for flags in flags_df.itertuples():
    file = flags[0]
    groundtruth = list(map(int, list(flags)[1].strip('['']').split(', ')))
    flag = np.array(flags[2:])

    pred = list(*np.where((flag != 0) & (flag > threshold)))
    if pred == []:
        sub = 0.0
        while pred == []:
            pred = list(*np.where((flag != 0) & (flag > threshold-sub)))
            sub += 0.1

    row = file + ':[' + ', '.join(map(str, groundtruth)) + ']:[' + ', '.join(map(str, pred)) + ']'
    row = row.split(':')

    with open('out_preds.csv', 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


## データの読み込み ##
filename = 'out_preds.csv'
preds_df = pd.read_csv(filename, usecols=['ファイル', '正解', 'pred'], index_col=0, encoding='Shift-JIS')


# 正しい指標

accs = []
for row in preds_df.itertuples():
    groundtruth = list(map(int, row[1].strip('['']').split(', ')))
    preds = list(map(int, row[2].strip('['']').split(', ')))
    
    correct = 0
    # 予測結果を1つずつ確認
    for pred in preds:
        # 予測結果が正解に含まれているか
        if pred in groundtruth:
            correct += 1
            
    out = len(list(set(groundtruth+preds)))
    accs.append(correct/out)

print('\n行動精度：', end='')
macro = np.mean(accs)
print(macro)




##** レシピ **#
label_num = 3


## データの読み込み ##
filename = 'Recipe_predictionleft_hip.tsv'
left_hip_df = pd.read_table(filename, usecols=['Filename', 'Groundtruth', 'EPOCH1000'], index_col=0, encoding='Shift-JIS')

filename = 'Recipe_predictionright_arm.tsv'
right_arm_df = pd.read_table(filename, usecols=['Filename', 'Groundtruth', 'EPOCH1000'], index_col=0, encoding='Shift-JIS')

filename = 'Recipe_predictionleft_wrist.tsv'
left_wrist_df = pd.read_table(filename, usecols=['Filename', 'Groundtruth', 'EPOCH1000'], index_col=0, encoding='Shift-JIS')

filename = 'Recipe_predictionright_wrist.tsv'
right_wrist_df = pd.read_table(filename, usecols=['Filename', 'Groundtruth', 'EPOCH1000'], index_col=0, encoding='Shift-JIS')




# ファイルの存在する部位だけ読み込み
# F値の比率計算
# フラグに掛け算，加算してファイルに書き出す
with open('out_flags_recipe.csv', 'w', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['ファイル', '正解', 'レシピ1', 'レシピ2', 'レシピ3'])


with open('labels.csv', 'r') as f:
    reader = csv.reader(f)
    all_files = [row[0] for row in reader]
all_files = [s for s in all_files_temp if s.startswith('subject2')]
    
    
for file in all_files:
    filename = file
    
    parts = ['left_hip', 'right_arm', 'left_wrist', 'right_wrist']
    preds = [[] for part in parts]

    f1_scores = np.zeros(4)
    f1_scores[0] = left_hip_df.at['Test Accuracy', 'EPOCH1000']
    f1_scores[1] = right_arm_df.at['Test Accuracy', 'EPOCH1000']
    f1_scores[2] = left_wrist_df.at['Test Accuracy', 'EPOCH1000']
    f1_scores[3] = right_wrist_df.at['Test Accuracy', 'EPOCH1000']
    
    
    isfile = left_hip_df.index.str.endswith(filename)
    if True in isfile and left_hip_df.at[filename, 'EPOCH1000'] != '[]':
        groundtruth = int(left_hip_df.at[filename, 'Groundtruth'])
        preds[0] = int(left_hip_df.at[filename, 'EPOCH1000'])
    else:
        f1_scores[0] = 0
    
    isfile = right_arm_df.index.str.endswith(filename)
    if True in isfile and right_arm_df.at[filename, 'EPOCH1000'] != '[]':
        groundtruth = int(right_arm_df.at[filename, 'Groundtruth'])
        preds[1] = int(right_arm_df.at[filename, 'EPOCH1000'])
    else:
        f1_scores[1] = 0
    
    isfile = left_wrist_df.index.str.endswith(filename)
    if True in isfile and left_wrist_df.at[filename, 'EPOCH1000'] != '[]':
        groundtruth = int(left_wrist_df.at[filename, 'Groundtruth'])
        preds[2] = int(left_wrist_df.at[filename, 'EPOCH1000'])
    else:
        f1_scores[2] = 0
    
    isfile = right_wrist_df.index.str.endswith(filename)
    if True in isfile and right_wrist_df.at[filename, 'EPOCH1000'] != '[]':
        groundtruth = int(right_wrist_df.at[filename, 'Groundtruth'])
        preds[3] = int(right_wrist_df.at[filename, 'EPOCH1000'])
    else:
        f1_scores[3] = 0

    if all([i == 0 for i in f1_scores]):
        continue    
    
    
    flags = [0 for i in range(label_num)]
    f1_ratio = f1_scores / max(f1_scores)
    
    for part, pred in enumerate(preds):
        if pred == []:
            continue
    
        else:
            flags[pred] += f1_ratio[part]
    
    row = flags
    row.insert(0, groundtruth)
    row.insert(0, filename)
    with open('out_flags_recipe.csv', 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)
    

# フラグのデータを読み出す
filename = 'out_flags_recipe.csv'
flags_df = pd.read_csv(filename, index_col=0, encoding='Shift-JIS')


# 最大になる閾値の計算
thresholds = np.arange(0.5, 3.5, 0.01)
correct = []
for threshold in thresholds:
    num = 0
    for flags in flags_df.itertuples():
        file = flags[0]
        groundtruth = flags[1]
        flag = np.array(flags[2:])
    
        pred = np.argmax(flag)

        if groundtruth == pred:
            num += 1
    correct.append(num)
print('\n完全正解：', end='')
print(max(correct))
threshold = thresholds[correct.index(max(correct))]


# 閾値最大で予測データのフラグをラベルに戻す
# 結果を書き出す
with open('out_preds_recipe.csv', 'w', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['ファイル', '正解', 'pred'])

for flags in flags_df.itertuples():
    file = flags[0]
    groundtruth = flags[1]
    flag = np.array(flags[2:])

    pred = np.argmax(flag)

    row = file + ':' + str(groundtruth) + ':' + str(pred)
    row = row.split(':')

    with open('out_preds_recipe.csv', 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

print('\nレシピ精度：', end='')
micro = max(correct)/len(flags_df)
print(micro)

print('\n***平均精度：', end='')
print((macro+micro)/2)







###*** シグモイド関数の生データに重みを掛ける手法 ***###
print("\n--手法2--\n")

##** 行動 **##
label_num = 10

## データの読み込み ##
filename = 'Activity_sigmoidleft_hip.tsv'
left_hip_df = pd.read_table(filename, usecols=['Filename', 'Groundtruth', 'EPOCH1000'], index_col=0, encoding='Shift-JIS')

filename = 'Activity_sigmoidright_arm.tsv'
right_arm_df = pd.read_table(filename, usecols=['Filename', 'Groundtruth', 'EPOCH1000'], index_col=0, encoding='Shift-JIS')

filename = 'Activity_sigmoidleft_wrist.tsv'
left_wrist_df = pd.read_table(filename, usecols=['Filename', 'Groundtruth', 'EPOCH1000'], index_col=0, encoding='Shift-JIS')

filename = 'Activity_sigmoidright_wrist.tsv'
right_wrist_df = pd.read_table(filename, usecols=['Filename', 'Groundtruth', 'EPOCH1000'], index_col=0, encoding='Shift-JIS')


# ファイルの存在する部位だけ読み込み
# F値の比率計算
# フラグに掛け算，加算してファイルに書き出す
with open('out_flags_sigmoid.csv', 'w', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['ファイル', '正解', '動作1', '動作2', '動作3', '動作4', '動作5', '動作6', '動作7', '動作8', '動作9', '動作10'])


with open('labels.csv', 'r') as f:
    reader = csv.reader(f)
    all_files = [row[0] for row in reader]
all_files = [s for s in all_files_temp if s.startswith('subject2')]
    
    
for file in all_files:
    filename = file
    
    parts = ['left_hip', 'right_arm', 'left_wrist', 'right_wrist']
    preds = [[] for part in parts]

    f1_scores = np.zeros(4)
    f1_scores[0] = left_hip_df.at['Test Accuracy', 'EPOCH1000']
    f1_scores[1] = right_arm_df.at['Test Accuracy', 'EPOCH1000']
    f1_scores[2] = left_wrist_df.at['Test Accuracy', 'EPOCH1000']
    f1_scores[3] = right_wrist_df.at['Test Accuracy', 'EPOCH1000']
    
    
    isfile = left_hip_df.index.str.endswith(filename)
    if True in isfile:
        groundtruth = list(map(int, left_hip_df.at[filename, 'Groundtruth'].strip('['']').split(', ')))
        preds[0] = list(map(float, left_hip_df.at[filename, 'EPOCH1000'].strip('['']').split(', ')))
    else:
        f1_scores[0] = 0
    
    isfile = right_arm_df.index.str.endswith(filename)
    if True in isfile:
        groundtruth = list(map(int, right_arm_df.at[filename, 'Groundtruth'].strip('['']').split(', ')))
        preds[1] = list(map(float, right_arm_df.at[filename, 'EPOCH1000'].strip('['']').split(', ')))
    else:
        f1_scores[1] = 0
    
    isfile = left_wrist_df.index.str.endswith(filename)
    if True in isfile:
        groundtruth = list(map(int, left_wrist_df.at[filename, 'Groundtruth'].strip('['']').split(', ')))
        preds[2] = list(map(float, left_wrist_df.at[filename, 'EPOCH1000'].strip('['']').split(', ')))
    else:
        f1_scores[2] = 0
    
    isfile = right_wrist_df.index.str.endswith(filename)
    if True in isfile:
        groundtruth = list(map(int, right_wrist_df.at[filename, 'Groundtruth'].strip('['']').split(', ')))
        preds[3] = list(map(float, right_wrist_df.at[filename, 'EPOCH1000'].strip('['']').split(', ')))
    else:
        f1_scores[3] = 0
    
    if all([i == 0 for i in f1_scores]):
        continue

    
    flags = [0 for i in range(label_num)]
    f1_ratio = f1_scores / max(f1_scores)
    
    for part, pred in enumerate(preds):
        if pred != []:
            flags += (np.array(pred)*f1_ratio[part])
    
    row = list(flags)
    row.insert(0, groundtruth)
    row.insert(0, filename)
    with open('out_flags_sigmoid.csv', 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)
    

# フラグのデータを読み出す
filename = 'out_flags_sigmoid.csv'
flags_df = pd.read_csv(filename, index_col=0, encoding='Shift-JIS')


# 最大になる閾値の計算
thresholds = np.arange(0.01, 3.00, 0.01)
correct = []
for threshold in thresholds:
    num = 0
    for flags in flags_df.itertuples():
        file = flags[0]
        groundtruth = list(map(int, list(flags)[1].strip('['']').split(', ')))
        flag = np.array(flags[2:])
    
        pred = list(*np.where((flag != 0) & (flag > threshold)))
        if pred == []:
            sub = 0.0
            while pred == []:
                pred = list(*np.where((flag != 0) & (flag > threshold-sub)))
                sub += 0.1

        if groundtruth == pred:
            num += 1
    correct.append(num)
print('\n完全正解：', end='')
print(max(correct))
threshold = thresholds[correct.index(max(correct))]


# 閾値最大で予測データのフラグをラベルに戻す
# 結果を書き出す
with open('out_preds_sigmoid.csv', 'w', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['ファイル', '正解', 'pred'])

for flags in flags_df.itertuples():
    file = flags[0]
    groundtruth = list(map(int, list(flags)[1].strip('['']').split(', ')))
    flag = np.array(flags[2:])

    pred = list(*np.where((flag != 0) & (flag > threshold)))
    if pred == []:
        sub = 0.0
        while pred == []:
            pred = list(*np.where((flag != 0) & (flag > threshold-sub)))
            sub += 0.1

    row = file + ':[' + ', '.join(map(str, groundtruth)) + ']:[' + ', '.join(map(str, pred)) + ']'
    row = row.split(':')

    with open('out_preds_sigmoid.csv', 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


## データの読み込み ##
filename = 'out_preds_sigmoid.csv'
preds_df = pd.read_csv(filename, usecols=['ファイル', '正解', 'pred'], index_col=0, encoding='Shift-JIS')



# 正しい指標

accs = []
for row in preds_df.itertuples():
    groundtruth = list(map(int, row[1].strip('['']').split(', ')))
    preds = list(map(int, row[2].strip('['']').split(', ')))
    
    correct = 0
    # 予測結果を1つずつ確認
    for pred in preds:
        # 予測結果が正解に含まれているか
        if pred in groundtruth:
            correct += 1
            
    out = len(list(set(groundtruth+preds)))
    accs.append(correct/out)

print('\n行動精度：', end='')
macro = np.mean(accs)
print(macro)



##** レシピ **##
label_num = 3


## データの読み込み ##
filename = 'Recipe_sigmoidleft_hip.tsv'
left_hip_df = pd.read_table(filename, usecols=['Filename', 'Groundtruth', 'EPOCH1000'], index_col=0, encoding='Shift-JIS')

filename = 'Recipe_sigmoidright_arm.tsv'
right_arm_df = pd.read_table(filename, usecols=['Filename', 'Groundtruth', 'EPOCH1000'], index_col=0, encoding='Shift-JIS')

filename = 'Recipe_sigmoidleft_wrist.tsv'
left_wrist_df = pd.read_table(filename, usecols=['Filename', 'Groundtruth', 'EPOCH1000'], index_col=0, encoding='Shift-JIS')

filename = 'Recipe_sigmoidright_wrist.tsv'
right_wrist_df = pd.read_table(filename, usecols=['Filename', 'Groundtruth', 'EPOCH1000'], index_col=0, encoding='Shift-JIS')




# ファイルの存在する部位だけ読み込み
# F値の比率計算
# フラグに掛け算，加算してファイルに書き出す
with open('out_flags_recipe_sigmoid.csv', 'w', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['ファイル', '正解', 'レシピ1', 'レシピ2', 'レシピ3'])


with open('labels.csv', 'r') as f:
    reader = csv.reader(f)
    all_files = [row[0] for row in reader]
all_files = [s for s in all_files_temp if s.startswith('subject2')]
    
    
for file in all_files:
    filename = file
    
    parts = ['left_hip', 'right_arm', 'left_wrist', 'right_wrist']
    preds = [[] for part in parts]

    f1_scores = np.zeros(4)
    f1_scores[0] = left_hip_df.at['Test Accuracy', 'EPOCH1000']
    f1_scores[1] = right_arm_df.at['Test Accuracy', 'EPOCH1000']
    f1_scores[2] = left_wrist_df.at['Test Accuracy', 'EPOCH1000']
    f1_scores[3] = right_wrist_df.at['Test Accuracy', 'EPOCH1000']
    
    
    isfile = left_hip_df.index.str.endswith(filename)
    if True in isfile:
        groundtruth = int(left_hip_df.at[filename, 'Groundtruth'])
        preds[0] = list(map(float, left_hip_df.at[filename, 'EPOCH1000'].strip('['']').split(', ')))
    else:
        f1_scores[0] = 0
    
    isfile = right_arm_df.index.str.endswith(filename)
    if True in isfile:
        groundtruth = int(right_arm_df.at[filename, 'Groundtruth'])
        preds[1] = list(map(float, right_arm_df.at[filename, 'EPOCH1000'].strip('['']').split(', ')))
    else:
        f1_scores[1] = 0
    
    isfile = left_wrist_df.index.str.endswith(filename)
    if True in isfile:
        groundtruth = int(left_wrist_df.at[filename, 'Groundtruth'])
        preds[2] = list(map(float, left_wrist_df.at[filename, 'EPOCH1000'].strip('['']').split(', ')))
    else:
        f1_scores[2] = 0
    
    isfile = right_wrist_df.index.str.endswith(filename)
    if True in isfile:
        groundtruth = int(right_wrist_df.at[filename, 'Groundtruth'])
        preds[3] = list(map(float, right_wrist_df.at[filename, 'EPOCH1000'].strip('['']').split(', ')))
    else:
        f1_scores[3] = 0
    
    if all([i == 0 for i in f1_scores]):
        continue
    
    
    flags = [0 for i in range(label_num)]
    f1_ratio = f1_scores / max(f1_scores)
    
    for part, pred in enumerate(preds):
        if pred != []:
            flags += (np.array(pred)*f1_ratio[part])

    
    row = list(flags)
    row.insert(0, groundtruth)
    row.insert(0, filename)
    with open('out_flags_recipe_sigmoid.csv', 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)
    

# フラグのデータを読み出す
filename = 'out_flags_recipe_sigmoid.csv'
flags_df = pd.read_csv(filename, index_col=0, encoding='Shift-JIS')


# 最大になる閾値の計算
thresholds = np.arange(0.01, 3.00, 0.01)
correct = []
for threshold in thresholds:
    num = 0
    for flags in flags_df.itertuples():
        file = flags[0]
        groundtruth = flags[1]
        flag = np.array(flags[2:])
    
        pred = np.argmax(flag)

        if groundtruth == pred:
            num += 1
    correct.append(num)
print('\n完全正解：', end='')
print(max(correct))
threshold = thresholds[correct.index(max(correct))]


# 閾値最大で予測データのフラグをラベルに戻す
# 結果を書き出す
with open('out_preds_recipe.csv', 'w', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['ファイル', '正解', 'pred'])

for flags in flags_df.itertuples():
    file = flags[0]
    groundtruth = flags[1]
    flag = np.array(flags[2:])

    pred = np.argmax(flag)

    row = file + ':' + str(groundtruth) + ':' + str(pred)
    row = row.split(':')

    with open('out_preds_recipe.csv', 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

print('\nレシピ精度：', end='')
micro = max(correct)/len(flags_df)
print(micro)

print('\n***平均精度：', end='')
print((macro+micro)/2)



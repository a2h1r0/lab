import numpy as np
import pandas as pd
import csv


### 本番 ###
##** レシピ **##
label_num = 3
recipes = ["sandwich", "fruitsalad", "cereal"]

## データの読み込み ##
filename = 'sigmoidleft_hip.tsv'
left_hip_df = pd.read_table(filename, usecols=['Filename', 'EPOCH1000'], index_col=0, encoding='Shift-JIS')

filename = 'sigmoidright_arm.tsv'
right_arm_df = pd.read_table(filename, usecols=['Filename', 'EPOCH1000'], index_col=0, encoding='Shift-JIS')

filename = 'sigmoidleft_wrist.tsv'
left_wrist_df = pd.read_table(filename, usecols=['Filename', 'EPOCH1000'], index_col=0, encoding='Shift-JIS')

filename = 'sigmoidright_wrist.tsv'
right_wrist_df = pd.read_table(filename, usecols=['Filename', 'EPOCH1000'], index_col=0, encoding='Shift-JIS')


files = []

for row in left_hip_df.itertuples():
    if row[0] == 'Total Train loss':
        break
    files.append(row[0])

for row in right_arm_df.itertuples():
    if row[0] == 'Total Train loss':
        break
    files.append(row[0])
    
for row in left_wrist_df.itertuples():
    if row[0] == 'Total Train loss':
        break
    files.append(row[0])

for row in right_wrist_df.itertuples():
    if row[0] == 'Total Train loss':
        break
    files.append(row[0])

all_files = list(set(files))


# ファイルの存在する部位だけ読み込み
# F値の比率計算
# フラグに掛け算，加算してファイルに書き出す
with open('out_flags_recipe_sigmoid.csv', 'w', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['ファイル', 'レシピ1', 'レシピ2', 'レシピ3'])
    
    
for file in all_files:
    filename = file
    
    parts = ['left_hip', 'right_arm', 'left_wrist', 'right_wrist']
    preds = [[] for part in parts]
    
    
    isfile = left_hip_df.index.str.endswith(filename)
    if True in isfile:
        preds[0] = list(map(float, left_hip_df.at[filename, 'EPOCH1000'].strip('['']').split(', ')))
    
    isfile = right_arm_df.index.str.endswith(filename)
    if True in isfile:
        preds[1] = list(map(float, right_arm_df.at[filename, 'EPOCH1000'].strip('['']').split(', ')))
    
    isfile = left_wrist_df.index.str.endswith(filename)
    if True in isfile:
        preds[2] = list(map(float, left_wrist_df.at[filename, 'EPOCH1000'].strip('['']').split(', ')))
    
    isfile = right_wrist_df.index.str.endswith(filename)
    if True in isfile:
        preds[3] = list(map(float, right_wrist_df.at[filename, 'EPOCH1000'].strip('['']').split(', ')))
    
    
    
    flags = [0 for i in range(label_num)]
    
    for pred in preds:
        if pred != []:
            flags += np.array(pred)

    
    row = list(flags)
    row.insert(0, filename)
    with open('out_flags_recipe_sigmoid.csv', 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)
    

# フラグのデータを読み出す
filename = 'out_flags_recipe_sigmoid.csv'
flags_df = pd.read_csv(filename, index_col=0, encoding='Shift-JIS')

# 結果を書き出す
with open('out_preds_recipe.csv', 'w', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['ファイル', 'pred'])

for flags in flags_df.itertuples():
    file = flags[0]
    flag = np.array(flags[1:])


    pred = np.argmax(flag)

        
    row = file + ',' + recipes[pred]
    row = row.split(',')

    with open('out_preds_recipe.csv', 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)






"""

### テスト ###

##** レシピ **##
label_num = 3
recipes = ["sandwich", "fruitsalad", "cereal"]

## データの読み込み ##
filename = 'Recipe_sigmoidleft_hip.tsv'
left_hip_df = pd.read_table(filename, usecols=['Filename', 'EPOCH1000'], index_col=0, encoding='Shift-JIS')

filename = 'Recipe_sigmoidright_arm.tsv'
right_arm_df = pd.read_table(filename, usecols=['Filename', 'EPOCH1000'], index_col=0, encoding='Shift-JIS')

filename = 'Recipe_sigmoidleft_wrist.tsv'
left_wrist_df = pd.read_table(filename, usecols=['Filename', 'EPOCH1000'], index_col=0, encoding='Shift-JIS')

filename = 'Recipe_sigmoidright_wrist.tsv'
right_wrist_df = pd.read_table(filename, usecols=['Filename', 'EPOCH1000'], index_col=0, encoding='Shift-JIS')




# ファイルの存在する部位だけ読み込み
# F値の比率計算
# フラグに掛け算，加算してファイルに書き出す
with open('out_flags_recipe_sigmoid.csv', 'w', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['ファイル', 'レシピ1', 'レシピ2', 'レシピ3'])


with open('labels.csv', 'r') as f:
    reader = csv.reader(f)
    all_files = [row[0] for row in reader]
    
    
for file in all_files:
    filename = file
    
    parts = ['left_hip', 'right_arm', 'left_wrist', 'right_wrist']
    preds = [[] for part in parts]
    
    
    isfile = left_hip_df.index.str.endswith(filename)
    if True in isfile:
        preds[0] = list(map(float, left_hip_df.at[filename, 'EPOCH1000'].strip('['']').split(', ')))
    
    isfile = right_arm_df.index.str.endswith(filename)
    if True in isfile:
        preds[1] = list(map(float, right_arm_df.at[filename, 'EPOCH1000'].strip('['']').split(', ')))
    
    isfile = left_wrist_df.index.str.endswith(filename)
    if True in isfile:
        preds[2] = list(map(float, left_wrist_df.at[filename, 'EPOCH1000'].strip('['']').split(', ')))
    
    isfile = right_wrist_df.index.str.endswith(filename)
    if True in isfile:
        preds[3] = list(map(float, right_wrist_df.at[filename, 'EPOCH1000'].strip('['']').split(', ')))
    
    
    
    flags = [0 for i in range(label_num)]
    
    for pred in preds:
        if pred != []:
            flags += np.array(pred)

    
    row = list(flags)
    row.insert(0, filename)
    with open('out_flags_recipe_sigmoid.csv', 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)
    

# フラグのデータを読み出す
filename = 'out_flags_recipe_sigmoid.csv'
flags_df = pd.read_csv(filename, index_col=0, encoding='Shift-JIS')

# 結果を書き出す
with open('out_preds_recipe.csv', 'w', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['ファイル', 'pred'])

for flags in flags_df.itertuples():
    file = flags[0]
    flag = np.array(flags[1:])


    pred = np.argmax(flag)

        
    row = file + ',' + recipes[pred]
    row = row.split(',')

    with open('preds_recipe_sigmoid.csv', 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)
"""
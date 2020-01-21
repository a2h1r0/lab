###--- ベクトルの平均値を計算 ---###
"""
ベクトルの平均値を計算．
被験者ごとにファイルからデータを読み込み，各取得回ごとの平均値を計算．
tester = [被験者名のリスト]を引数に取り，
vector_ave[被験者] = [[ベクトルのリスト] * 取得回]を返す．
"""


def calculate_vector_ave(tester):
    import pandas as pd
    import numpy as np

    vector_ave = [[] for i in tester]   # 結果用

    ## データの読み込み ##
    for order, name in enumerate(tester):   # 被験者1人ずつデータを読み込む
        filename = "./dataset/" + name + ".csv"
        data = pd.read_csv(filename, usecols=["in0","in1","in2","in3","in4","in5","in6","in7",
                                              "in8","in9","inあ","inい","inう","inA","inB","inC",
                                              "in10","in11","in12","in13","in14","in15","in16",
                                              "in17","in18","in19","inア","inイ","inウ","inD","inE",
                                              "inF","Number"], encoding='Shift-JIS')
        data.fillna(0, inplace=True)    # 区切り番号以外"0"で埋める
        sensors = len(data.iloc[0])-1   # 次元数の確認
        
        ## データの計算 ## : 各データ，各取得回の区切りごとに平均値を計算
        # 被験者変更時に変数を初期化
        vector_temp = np.zeros(sensors)     # ベクトルの平均値計算用
        num = 0     # データ数(計算回数)
    
        # 平均値の計算
        for row in data.itertuples(name=None):   ## 1行ずつ読み出し
            row = list(row)
            # 区切りごとに平均値を保存，変数を初期化
            if not row[-1] in (0, 1):   # 区切りの検出，最初の区切り"1"ではスキップ
                vector_ave[order].append(vector_temp/num)   # 平均値を計算
                vector_temp = np.zeros(sensors)             # 計算用変数の初期化
                num = 0
                
            del row[-1]     # 末尾は区切り文字
            del row[0]      # 先頭はデータ番号
            vector_temp += row  # 1行ずつ加算していく
            num += 1            # データ数(計算回数)を増加
            
        # 最終データの平均値を保存
        vector_ave[order].append(vector_temp/num)
        
    return vector_ave
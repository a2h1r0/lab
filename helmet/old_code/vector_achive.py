import os.path
import csv
import pandas as pd
import sys

tester = ["ooyama", "okamoto", "kajiwara", "sawano", "nagamatsu", "noda", "hatta", "matsuda"]
writefile = "testdata.csv"

if os.path.isfile(writefile):    # データ保存先の存在確認
    print("ファイルが存在します．\n")
    sys.exit()

    
with open(writefile, 'a', newline='') as f:  # 保存先をオープン
    writer = csv.writer(f)

    ## ラベルの付与と取得回数の確認 ##
    writer.writerow(["in0","in1","in2","in3","in4","in5","in6","in7",
                     "in8","in9","inあ","inい","inう","inA","inB","inC",
                     "in10","in11","in12","in13","in14","in15","in16",
                     "in17","in18","in19","inア","inイ","inウ","inD","inE",
                     "inF","Tester"])

    for name in tester:
        ## データの読み込み ##
        readfile = name + "_ave.csv"
        data = [] # データ配列，被験者数分用意
        data = pd.read_csv(readfile, usecols=["in0","in1","in2","in3","in4","in5","in6","in7",
                                                "in8","in9","inあ","inい","inう","inA","inB","inC",
                                                "in10","in11","in12","in13","in14","in15","in16",
                                                "in17","in18","in19","inア","inイ","inウ","inD","inE",
                                                "inF"], encoding='Shift-JIS')

        for row in data.itertuples(name=None):   ## 1行ずつ読み出し:
            row = list(row)
            del row[0]
            row.append(name)
            writer.writerow(row)

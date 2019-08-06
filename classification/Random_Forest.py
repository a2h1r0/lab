import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RFC

# CSVファイルの読み込み
train = pd.read_csv('train.csv') # 学習データ
test = pd.read_csv('test.csv')   # 評価データ

# 学習データの説明変数
train_data = train[["V0","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10",
                    "V11","V12","V13","V14","V15","V16","V17","V18","V19"]]
# 学習データの目的変数
train_label = pd.get_dummies(train["Tester"])
# 評価データの説明変数
test_data = test[["V0","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10",
                  "V11","V12","V13","V14","V15","V16","V17","V18","V19"]]
# 評価データの目的変数
test_label = pd.get_dummies(test["Tester"])

clf = RFC(random_state=42)        # ランダムフォレストを選択
clf.fit(train_data, train_label)  # 学習

print(f"acc: {clf.score(test_data, test_label)}") # 予測
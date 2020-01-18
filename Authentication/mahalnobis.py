###--- データにより随時変更 ---###
tester = ["ooyama", "okamoto", "kajiwara", "sawano", "nagamatsu", "noda", "hatta", "fujii", "matsuda"]  # **被験者**
MIN = 0  # **閾値の下限**
MAX = 600  # **閾値の上限**
digit = 1  # **桁数調整**(閾値に小数を用いる場合，1桁ごとに10倍)
k = 5  # **交差検証分割数**
###--- ここまで ---###


import numpy as np
import calculate_vector_ave as cal
from sklearn.covariance import MinCovDet
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")

###--- データの作成関数 ---###
"""
学習データと認証データを作成．
学習データは交差検証で本人のテストデータに指定された部分を除いて作成．
認証データは先頭に本人のテストデータを格納し，後ろに他の被験者データを格納．
"""


def make_testdata():
    ## 学習データの作成　# 交差検証で指定されたテストデータを除く
    for i in range(k):
        if i != order:
            train_data.extend(vector_ave[index_train][i * data_size:(i + 1) * data_size])

    ## 認証(正解)データの作成　# 本人のテストデータを先頭に格納
    attack_data.extend(vector_ave[index_train][order * data_size:(order + 1) * data_size])
    ## 認証(異常)データの追加　# 本人以外のテストデータを追加
    for index_attack in range(len(tester)):
        if index_attack != index_train:
            attack_data.extend(vector_ave[index_attack])


###--- ここまで ---###

###--- データの判別関数 ---###
"""
マハラノビス距離と閾値でデータを判別．
認証データの先頭からdata_size番目までは本人のデータ，そこから他人のデータが格納．
FRR(FAR)[学習中の被験者][閾値の要素番号]を作成．
"""
thresholds = np.linspace(MIN, MAX, int((MAX - MIN) * digit + 1))  # 閾値の配列
FRR = np.zeros((len(tester), len(thresholds)))  # 結果用配列
FAR = np.zeros((len(tester), len(thresholds)))


def compare():
    FRR_num = np.zeros(len(thresholds))  # 計算用配列
    FAR_num = np.zeros(len(thresholds))
    for index, threshold in enumerate(thresholds):  # 閾値移動
        for item, distance in enumerate(score):  # 値を1つずつ取り出す
            if item < data_size and distance > threshold:  # scoreのdata_size番目までは正解データ
                FRR_num[index] += 1
            elif item >= data_size and distance <= threshold:  # それ以降は異常データ
                FAR_num[index] += 1

    # 被験者ごとに結果を保存
    FRR[index_train] += FRR_num / data_size  # 閾値ごとに[data_size]回だけ正解データと比較
    FAR[index_train] += FAR_num / (len(score) - data_size)  # 閾値ごとに[全体のデータ数-正解データ数]回だけ異常データと比較


###--- ここまで ---###


###--- main ---###
## 計算と判別 ##
vector_ave = cal.calculate_vector_ave(tester)  # ベクトルの平均値を計算

mcd = MinCovDet()  # Minimum Covariance Determinant
for index_train in range(len(tester)):  ## 学習する被験者を変更
    data_size = int(len(vector_ave[index_train]) / k)  # データサイズの計算

    for order in range(k):  ## 交差検証，テストデータを選択
        train_data = []  # データセットの初期化
        attack_data = []
        make_testdata()  # データセットの作成
        mcd.fit(train_data)  # 学習
        score = mcd.mahalanobis(attack_data)  # マハラノビス距離を計算
        compare()  # 判別

    FRR[index_train] /= k  # 結果を交差検証の試行回数で除算
    FAR[index_train] /= k
FRR *= 100  # 全体を百分率化
FAR *= 100

FRR_total = FRR.mean(axis=0)  # 全ての被験者での平均値を計算
FAR_total = FAR.mean(axis=0)  # 被験者ごとの行，閾値の列になっているので，列の平均値

"""
## 結果の描画 ##
plt.figure(0)  # 複数ウィンドウで表示
plt.title("Total", fontsize=18)
plt.plot(thresholds, FRR_total, 'red', label="FRR")
plt.plot(thresholds, FAR_total, 'blue', linestyle="dashed", label="FAR")
plt.xlabel("Threshold", fontsize=18)
plt.ylabel("Rate", fontsize=18)
plt.tick_params(labelsize=18)
plt.legend(fontsize=18)  # 凡例の表示
#plt.savefig("EER_total.eps", bbox_inches='tight', pad_inches=0)
"""







#**論文用，一覧グラフ取得**
tester_index = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
plt.figure(figsize=(15, 30))
plt.subplots_adjust(wspace=0.4, hspace=0.4)
for index, name in enumerate(tester_index):
    if name == "E": # Eだけ除外
        continue
    plt.subplot(5, 2, index+1)
    plt.title("Subject "+name, fontsize=18)
    plt.plot(thresholds, FRR[index], 'red', label="FRR")
    plt.plot(thresholds, FAR[index], 'blue', linestyle="dashed", label="FAR")
    plt.xlabel("Threshold", fontsize=18)
    plt.ylabel("Rate", fontsize=18)
    plt.tick_params(labelsize=18)
    plt.legend(fontsize=18)  # 凡例の表示

plt.subplot(5, 2, 10)
plt.title("Total", fontsize=18)
plt.plot(thresholds, FRR_total, 'red', label="FRR")
plt.plot(thresholds, FAR_total, 'blue', linestyle="dashed", label="FAR")
plt.xlabel("Threshold", fontsize=18)
plt.ylabel("Rate", fontsize=18)
plt.tick_params(labelsize=18)
plt.legend(fontsize=18)  # 凡例の表示


## 被験者Eのために閾値を変更，再計算
MIN = 3000  # **閾値の下限**
MAX = 3600  # **閾値の上限**
thresholds = np.linspace(MIN, MAX, int((MAX - MIN) * digit + 1))  # 閾値の配列
FRR = np.zeros((len(tester), len(thresholds)))  # 結果用配列
FAR = np.zeros((len(tester), len(thresholds)))

vector_ave = cal.calculate_vector_ave(tester)  # ベクトルの平均値を計算

mcd = MinCovDet()  # Minimum Covariance Determinant
for index_train in range(len(tester)):  ## 学習する被験者を変更
    data_size = int(len(vector_ave[index_train]) / k)  # データサイズの計算

    for order in range(k):  ## 交差検証，テストデータを選択
        train_data = []  # データセットの初期化
        attack_data = []
        make_testdata()  # データセットの作成
        mcd.fit(train_data)  # 学習
        score = mcd.mahalanobis(attack_data)  # マハラノビス距離を計算
        compare()  # 判別

    FRR[index_train] /= k  # 結果を交差検証の試行回数で除算
    FAR[index_train] /= k
FRR *= 100  # 全体を百分率化
FAR *= 100

FRR_total = FRR.mean(axis=0)  # 全ての被験者での平均値を計算
FAR_total = FAR.mean(axis=0)  # 被験者ごとの行，閾値の列になっているので，列の平均値


## 描画
plt.subplot(5, 2, 5)
plt.title("Subject E", fontsize=18)
plt.plot(thresholds, FRR[4], 'red', label="FRR")
plt.plot(thresholds, FAR[4], 'blue', linestyle="dashed", label="FAR")
plt.xlabel("Threshold", fontsize=18)
plt.ylabel("Rate", fontsize=18)
plt.tick_params(labelsize=18)
plt.legend(fontsize=18)  # 凡例の表示
plt.savefig("EER.eps", bbox_inches='tight', pad_inches=0)

plt.show()
#**ここまで**
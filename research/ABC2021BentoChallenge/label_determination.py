def majority_vote(predictions):
    """
    多数決

    Args:
        predictions (array): 予測
    Returns:
        int: 出力ラベル
    """

    sum_predictions = np.sum(predictions, axis=0)
    output = np.argmax(sum_predictions) + 1

    return output


def majority_vote_sig(predictions, threshold):
    """
    入力データがシグモイド関数に順する場合の多数決

    Args:
        predictions (array): 予測
        threshold (int): 閾値
    Returns:
        int: 出力ラベル
    """

    gtt_predictions = predictions

    # 閾値未満を0に置換
    for i in range(len(predictions)):
        for j in range(len(predictions[i])):
            if predictions[i][j] < threshold:
                gtt_predictions[i][j] = 0
            else:
                gtt_predictions[i][j] = predictions[i][j]

    sum_predictions = np.sum(gtt_predictions, axis=0)
    # 精度の合計が最大のインデックスを複数保存
    sum_max_index = [i for i, x in enumerate(sum_predictions) if x == max(sum_predictions)]

    # 合計が最大のインデックスが複数あった場合
    if len(sum_max_index) > 1:

        temp_predictions = np.array(predictions).T
        m = 0

        # 合計が最大のインデックスのデータのみ比較して、最大の要素が格納されているインデックスをmaxIndexに保存
        for i in range(len(sum_max_index)):
            if m < max(temp_predictions[sum_max_index[i]]):
                m = max(temp_predictions[sum_max_index[i]])
                maxIndex = sum_max_index[i]

        output = maxIndex + 1

    # ひとつだった場合
    else:
        output = sum_max_index[0] + 1

    return output


if __name__ == '__main__':
    import numpy as np

    right_shoulder_predictions = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    right_elbow_predictions = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    left_wrist_predictions = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

    predictions = [right_shoulder_predictions, right_elbow_predictions, left_wrist_predictions]

    prediction = majority_vote(predictions)

    print(prediction)

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


if __name__ == '__main__':
    import numpy as np

    right_shoulder_predictions = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    right_elbow_predictions = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    left_wrist_predictions = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

    predictions = [right_shoulder_predictions, right_elbow_predictions, left_wrist_predictions]

    prediction = majority_vote(predictions)

    print(prediction)

import time


class Exam():
    """
    被験者実験テスト
    """

    def calc(question_length):
        """
        Args:
            question_length (number): 設問数
        Returns:
            :obj:`Tensor`[batch_size, 1]: 識別結果
        """

        time.sleep(question_length)

import random


class Exam():
    """
    被験者実験テスト
    """

    def calc(question_length):
        """
        計算テスト

        Args:
            question_length (number): 設問数
        Returns:
            :obj:`Tensor`[batch_size, 1]: 識別結果
        """

        def make_questions(question_length):
            """
            計算問題の作成

            Args:
                question_length (number): 設問数
            Returns:
                list: 設問
                list: 正解
            """

            questions = [random.sample(range(10, 99), 2)
                         for i in range(question_length)]
            answers = [sum(question) for question in questions]

            return questions, answers

        questions, answers = make_questions(question_length)

        print(questions)
        print(answers)

        # todo: 設問表示，回答部分の実装

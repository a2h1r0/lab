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
            list: 設問回答データ
        """

        def make_questions(question_length):
            """
            計算問題の作成

            Args:
                question_length (number): 設問数
            Returns:
                list: 設問
            """

            questions = [random.sample(range(10, 99), 2)
                         for i in range(question_length)]

            return questions

        questions = make_questions(question_length)
        answers = [['answer', 'correct']]
        correct_count = 0

        for i, question in enumerate(questions):
            answer = input(f'\n({i + 1}) {question[0]} + {question[1]} = ')

            is_correct = int(answer) == sum(question)
            if is_correct:
                correct_count += 1

            answers.append(
                [f'{question[0]}+{question[1]}={answer}', is_correct])

        answers.append(['(Accuracy)', correct_count / (len(answers) - 1)])

        return answers

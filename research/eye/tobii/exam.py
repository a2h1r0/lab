import random
import utils
import cv2
import os


class Exam():
    """
    被験者実験テスト
    """

    def calc(question_length, digit):
        """
        計算テスト

        Args:
            question_length (number): 設問数
            digit (number): 問題作成桁数
        Returns:
            list: 設問回答データ
        """

        def make_questions(question_length, digit):
            """
            計算問題の作成

            Args:
                question_length (number): 設問数
                digit (number): 問題作成桁数
            Returns:
                list: 設問
            """

            questions = [random.sample(range(digit, digit * 10), 2)
                         for i in range(question_length)]

            return questions

        questions = make_questions(question_length, digit)
        answers = [['answer', 'correct']]
        correct_count = 0

        for i, question in enumerate(questions):
            answer = utils.input_decimal(
                f'\n({i + 1}) {question[0]} + {question[1]} = ')

            is_correct = answer == sum(question)
            if is_correct:
                correct_count += 1

            answers.append(
                [f'{question[0]}+{question[1]}={answer}', is_correct])

        answers.append(['(Accuracy)', correct_count / (len(answers) - 1)])

        return answers

    def look(points, waiting):
        """
        注視テスト

        Args:
            points (list): 描画座標一覧
            waiting (number): 静止時間（秒）
        """

        def draw_point(point, waiting):
            """
            注視ポイントの描画

            Args:
                point (tuple): 描画座標
                waiting (number): 静止時間（秒）
            """

            img = cv2.imread(
                f'{os.path.dirname(__file__)}/eyetracker/calibration/{point[0]}_{point[1]}.png')
            cv2.imshow('screen', img)

            cv2.waitKey(waiting * 1000)

        cv2.namedWindow('screen', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(
            'screen', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        for point in points:
            draw_point(point, waiting)

        cv2.destroyWindow('screen')

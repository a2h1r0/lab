import tobii_research as tr
import cv2
import os


class Tobii():
    """
    Tobii操作クラス
    """

    def __init__(self):
        self.eyetracker = self.connect_eyetracker()
        self.data = []

    def calibration(self):
        """
        キャリブレーション

        Returns:
            bool: 結果
        """

        def collect_data(calibration, points):
            """
            キャリブレーションデータの取得

            Args:
                calibration (:class:`ScreenBasedCalibration`): キャリブレーションクラス
                points (list): 描画ポイント
            Returns:
                list: 再描画ポイント
            """

            def draw_point(point):
                """
                キャリブレーションポイントの描画

                Args:
                    point (tuple): 描画座標
                """

                cv2.namedWindow('screen', cv2.WINDOW_NORMAL)
                cv2.setWindowProperty(
                    'screen', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

                img = cv2.imread(
                    f'{os.path.dirname(__file__)}/calibration/{point[0]}_{point[1]}.png')
                cv2.imshow('screen', img)

                # 視線移動待ち
                cv2.waitKey(1000)

            recalibrate = []
            for point in points:
                draw_point(point)

                status = calibration.collect_data(point[0], point[1])
                if status != tr.CALIBRATION_STATUS_SUCCESS:
                    status = calibration.collect_data(point[0], point[1])

                    if status != tr.CALIBRATION_STATUS_SUCCESS:
                        recalibrate.append(point)

            return recalibrate

        POINTS_TO_CALIBRATE = [(0.5, 0.5), (0.1, 0.1),
                               (0.1, 0.9), (0.9, 0.1), (0.9, 0.9)]

        calibration = tr.ScreenBasedCalibration(self.eyetracker)
        calibration.enter_calibration_mode()

        # キャリブレーション
        points_to_recalibrate = collect_data(calibration, POINTS_TO_CALIBRATE)

        # キャリブレーション再実行
        if len(points_to_recalibrate):
            collect_data(calibration, points_to_recalibrate)

        result = calibration.compute_and_apply()
        calibration.leave_calibration_mode()

        cv2.destroyWindow('screen')

        return len(result.calibration_points) == len(POINTS_TO_CALIBRATE)

    def connect_eyetracker(self):
        """
        Tobiiの接続

        Returns:
            :tuple:`EyeTracker`: 接続情報
        """

        return tr.find_all_eyetrackers()[0]

    def gaze_data_callback(self, data):
        """
        データの取得

        Args:
            data (dict): 取得データ
        """

        coordinates = ['x', 'y', 'z']

        keys, values = [], []
        for (key, value) in list(data.items()):
            if isinstance(value, tuple):
                keys.extend(
                    [f'{key}_{coordinates[i]}' for i in range(len(value))])
                values.extend(list(value))
            else:
                keys.append(key)
                values.append(value)

        if self.data == []:
            self.data.append(keys)
        self.data.append(values)

    def subscribe(self):
        """
        データの取得開始
        """

        self.eyetracker.subscribe_to(
            tr.EYETRACKER_GAZE_DATA, self.gaze_data_callback, as_dictionary=True)

    def unsubscribe(self):
        """
        データの取得終了
        """

        self.eyetracker.unsubscribe_from(
            tr.EYETRACKER_GAZE_DATA, self.gaze_data_callback)

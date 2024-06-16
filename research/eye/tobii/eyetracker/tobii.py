import tobii_research as tr
import time
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
        def draw_point(point):
            cv2.namedWindow('screen', cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(
                'screen', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            img = cv2.imread(
                f'{os.path.dirname(__file__)}/calibration/{point[0]}_{point[1]}.png')
            cv2.imshow('screen', img)

            # 注目待ち
            cv2.waitKey(700)

        POINTS_TO_CALIBRATE = [(0.5, 0.5), (0.1, 0.1),
                               (0.1, 0.9), (0.9, 0.1), (0.9, 0.9)]

        calibration = tr.ScreenBasedCalibration(self.eyetracker)
        calibration.enter_calibration_mode()

        points_to_recalibrate = []
        for point in POINTS_TO_CALIBRATE:
            draw_point(point)

            for failed in range(3):
                # キャリブレーション
                status = calibration.collect_data(point[0], point[1])
                if status == tr.CALIBRATION_STATUS_SUCCESS:
                    break
            if status != tr.CALIBRATION_STATUS_SUCCESS:
                points_to_recalibrate.append(point)

        # キャリブレーション適用
        calibration_result = calibration.compute_and_apply()
        print("Compute and apply returned {0} and collected at {1} points.".
              format(calibration_result.status, len(calibration_result.calibration_points)))

        if len(points_to_recalibrate):
            for recalibrate_point in points_to_recalibrate:
                calibration.discard_data(
                    recalibrate_point[0], recalibrate_point[1])

                draw_point(recalibrate_point)

                for failed in range(3):
                    # キャリブレーション
                    status = calibration.collect_data(
                        recalibrate_point[0], recalibrate_point[1])
                    if status == tr.CALIBRATION_STATUS_SUCCESS:
                        break

            # Compute and apply again.
            print("Computing and applying calibration.")
            calibration_result = calibration.compute_and_apply()
            print("Compute and apply returned {0} and collected at {1} points.".
                  format(calibration_result.status, len(calibration_result.calibration_points)))

        # See that you're happy with the result.

        # The calibration is done. Leave calibration mode.
        calibration.leave_calibration_mode()

        print("Left calibration mode.")
        # <EndExample>

    def connect_eyetracker(self):
        """
        Tobiiの接続

        Returns:
            :tuple:`EyeTracker`: 接続情報
        """

        return tr.find_all_eyetrackers()[0]

    def get_gaze_data(self, data):
        """
        データの取得

        Args:
            data (dict): 取得データ
        """

        keys, values = [], []
        for (key, value) in list(data.items()):
            if isinstance(value, tuple):
                keys.extend([f'{key}_{i + 1}' for i in range(len(value))])
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
            tr.EYETRACKER_GAZE_DATA, self.get_gaze_data, as_dictionary=True)

    def unsubscribe(self):
        """
        データの取得終了
        """

        self.eyetracker.unsubscribe_from(
            tr.EYETRACKER_GAZE_DATA, self.get_gaze_data)

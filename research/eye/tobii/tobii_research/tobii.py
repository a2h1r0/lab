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

            cv2.waitKey(2500)  # 2.5秒キャリブレーションする

        calibration = tr.ScreenBasedCalibration(self.eyetracker)

        # Enter calibration mode.
        calibration.enter_calibration_mode()
        print("Entered calibration mode for eye tracker with serial number {0}.".format(
            self.eyetracker.serial_number))

        # Define the points on screen we should calibrate at.
        # The coordinates are normalized, i.e. (0.0, 0.0) is the upper left corner and (1.0, 1.0) is the lower right corner.
        points_to_calibrate = [(0.5, 0.5), (0.1, 0.1),
                               (0.1, 0.9), (0.9, 0.1), (0.9, 0.9)]

        for point in points_to_calibrate:
            print("Show a point on screen at {0}.".format(point))
            draw_point(point)

            # # Wait a little for user to focus.
            # time.sleep(0.7)

            print("Collecting data at {0}.".format(point))
            if calibration.collect_data(point[0], point[1]) != tr.CALIBRATION_STATUS_SUCCESS:
                # Try again if it didn't go well the first time.
                # Not all eye tracker models will fail at this point, but instead fail on ComputeAndApply.
                calibration.collect_data(point[0], point[1])

        print("Computing and applying calibration.")
        calibration_result = calibration.compute_and_apply()
        print("Compute and apply returned {0} and collected at {1} points.".
              format(calibration_result.status, len(calibration_result.calibration_points)))

        # Analyze the data and maybe remove points that weren't good.
        recalibrate_point = (0.1, 0.1)
        print("Removing calibration point at {0}.".format(recalibrate_point))
        calibration.discard_data(recalibrate_point[0], recalibrate_point[1])

        # Redo collection at the discarded point
        print("Show a point on screen at {0}.".format(recalibrate_point))
        calibration.collect_data(recalibrate_point[0], recalibrate_point[1])

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

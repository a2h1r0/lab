import tobii_research as tr


class Tobii():
    """
    Tobii操作クラス
    """

    def __init__(self):
        self.eyetracker = self.connect_eyetracker()
        self.data = []

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

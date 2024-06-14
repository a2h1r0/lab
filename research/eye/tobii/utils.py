def input_decimal(message):
    """
    数値の入力

    Args:
        message (string): 表示メッセージ
    Returns:
        int: 入力データ
    """

    while True:
        decimal = input(message)

        if decimal.isdecimal():
            break
        else:
            print('数値を入力してください．\n')

    return int(decimal)

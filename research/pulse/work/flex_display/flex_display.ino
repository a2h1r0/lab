// 目標心拍数
const int H_TARGET = 50;

// 使用ピン
const int PWM = 3;
// 黒の電圧値 (V)
const int BLACK = 5;
// 白の電圧値 (V)
const int WHITE = 0;
// 色データ長
const int L_COLORS = 6;
// 色データ
const int COLORS[L_COLORS] = {WHITE, WHITE, BLACK / 2, BLACK, BLACK / 2, WHITE};
// 描画間隔
const int DELAY = 60 / (L_COLORS * H_TARGET);

/**
 * @fn
 * 初期化
 */
void setup()
{
    pinMode(PWM, OUTPUT);
}

/**
 * @fn
 * メインループ
 */
void loop()
{
    for (int i = 0; i < L_COLORS; i++)
    {
        analogWrite(PWM, COLORS[i] * 51);
        delay(DELAY);
    }
}

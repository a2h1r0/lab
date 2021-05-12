// 使用ピン
const int PWM = 3;
// 黒の電圧値 (V)
const int BLACK = 5;
// 白の電圧値 (V)
const int WHITE = 0;
// 描画間隔
const int DELAY = 500;

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
    analogWrite(PWM, BLACK * 51);
    delay(DELAY);
    analogWrite(PWM, WHITE * 51);
    delay(DELAY);
}

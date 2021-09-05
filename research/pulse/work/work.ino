// 目標心拍数
const int HEART_RATE = 80;

// 使用ピン
const int PWM_1 = 5;
const int PWM_2 = 6;
// 黒の電圧値 (V)
const int BLACK_PWM_1 = 2;
const int BLACK_PWM_2 = 0;
// 白の電圧値 (V)
const int WHITE_PWM_1 = 0;
const int WHITE_PWM_2 = 2;
// 色データ長
const int L_COLORS = 2;
// 色データ
const int COLORS_PWM_1[L_COLORS] = {WHITE_PWM_1, BLACK_PWM_1};
const int COLORS_PWM_2[L_COLORS] = {WHITE_PWM_2, BLACK_PWM_2};
// 目標心拍数受け取り用配列
char heart_rate_data[4] = {'\0'};

/**
 * @fn
 * 初期化
 */
void setup()
{
    pinMode(PWM_1, OUTPUT);
    pinMode(PWM_2, OUTPUT);
    Serial.begin(9600);
}

/**
 * @fn
 * メインループ
 */
void loop()
{
    // 点灯時間の計算
    float lighting_time = (float)60 / (L_COLORS * HEART_RATE);
    // ミリ秒変換
    int delay_time = lighting_time * 1000;

    // 描画
    while (true)
    {
        for (int i = 0; i < L_COLORS; i++)
        {
            analogWrite(PWM_1, COLORS_PWM_1[i] * 51);
            analogWrite(PWM_2, COLORS_PWM_2[i] * 51);
            delay(delay_time);
        }
    }
}

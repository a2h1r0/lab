// 目標心拍数
const int HEART_RATE = 80;

// 使用ピン
const int PWM_1 = 5;
const int PWM_2 = 6;
// 電圧値 (V)
const int PWM_HIGH = 2;
const int PWM_LOW = 0;
// 色データ長
const int L_COLORS = 9;
// 色データ
const int COLORS_PWM_1[L_COLORS] = {
    PWM_HIGH,
    PWM_LOW,
};
const int COLORS_PWM_2[L_COLORS] = {
    PWM_LOW,
    PWM_HIGH,
};

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

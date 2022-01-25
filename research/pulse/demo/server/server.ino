// 使用ピン
const int PWM_1 = 5;
const int PWM_2 = 6;
// 電圧値 (V)
const int PWM_HIGH = 2;
const int PWM_LOW = 0;
// 色データ長
const int L_COLORS = 2;
// 色データ
const int COLORS_PWM_1[L_COLORS] = {
    PWM_HIGH,
    PWM_LOW,
};
const int COLORS_PWM_2[L_COLORS] = {
    PWM_LOW,
    PWM_HIGH,
};
// 目標心拍数受け取り用配列
char heart_rate[4] = {'\0'};
// 点灯時間
int delay_time = 500;

void setup()
{
    pinMode(PWM_1, OUTPUT);
    pinMode(PWM_2, OUTPUT);
    Serial.begin(9600);
}

/**
 * @fn
 * 点灯時間の計算
 */
int get_delay_time(int heart_rate)
{
    // 点灯時間の計算
    float lighting_time = (float)60 / (L_COLORS * heart_rate);

    // ミリ秒変換
    return (int)(lighting_time * 1000);
}

void loop()
{
    if (Serial.available())
    {
        // 目標心拍数の取得
        Serial.readStringUntil('\0').toCharArray(heart_rate, sizeof heart_rate);
        // 点灯時間の更新
        delay_time = get_delay_time(atoi(heart_rate));

        // 目標心拍数更新処理の終了通知
        Serial.println(0);
    }

    // 描画
    for (int i = 0; i < L_COLORS; i++)
    {
        analogWrite(PWM_1, COLORS_PWM_1[i] * 51);
        analogWrite(PWM_2, COLORS_PWM_2[i] * 51);
        delay(delay_time);
    }
}

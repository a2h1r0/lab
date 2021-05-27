// 実行時間
const int PROCESS_TIME = 130;

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
    if (Serial.available() > 0)
    {
        // 目標心拍数の取得
        int heart_rate = Serial.readStringUntil('\0');
        // 点灯時間の計算
        int lighting_time = 60 / (L_COLORS * heart_rate);

        // 描画
        for (int i = 0; i < L_COLORS; i++)
        {
            analogWrite(PWM, COLORS[i] * 51);
            delay(lighting_time);
        }

        // 描画完了通知の送信
        Serial.println(0);
    }
}

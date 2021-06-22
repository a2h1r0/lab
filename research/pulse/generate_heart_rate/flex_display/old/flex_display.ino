// 実行時間
const int PROCESS_TIME = 40;

// 使用ピン
const int PWM_1 = 5;
const int PWM_2 = 6;
// 黒の電圧値 (V)
const int BLACK_PWM_1 = 5;
const int BLACK_PWM_2 = 0;
// 白の電圧値 (V)
const int WHITE_PWM_1 = 0;
const int WHITE_PWM_2 = 5;
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
    // 黒くした状態で待機
    analogWrite(PWM_1, BLACK_PWM_1 * 51);
    analogWrite(PWM_2, BLACK_PWM_2 * 51);

    if (Serial.available())
    {
        // 目標心拍数の取得
        Serial.readStringUntil('\0').toCharArray(heart_rate_data, sizeof heart_rate_data);
        int heart_rate = atoi(heart_rate_data);
        // 点灯時間の計算
        float lighting_time = (float)60 / (L_COLORS * heart_rate);
        // ミリ秒変換
        int delay_time = lighting_time * 1000;

        // 描画開始時間の取得
        long start = micros();

        // 描画
        while (true)
        {
            // 時間経過で終了
            long process = micros() - start;
            if (process > (PROCESS_TIME * 1000000))
            {
                break;
            }

            for (int i = 0; i < L_COLORS; i++)
            {
                analogWrite(PWM_1, COLORS_PWM_1[i] * 51);
                analogWrite(PWM_2, COLORS_PWM_2[i] * 51);
                delay(delay_time);
            }
        }

        // 描画完了通知の送信
        Serial.println(0);
    }
}

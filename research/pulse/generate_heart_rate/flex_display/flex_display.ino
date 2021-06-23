#include <cppQueue.h>

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
int i = 0;
unsigned long drew_time;
cppQueue peaks(sizeof(unsigned long), 5);

/**
 * 初期化
 */
void setup()
{
    pinMode(PWM_1, OUTPUT);
    pinMode(PWM_2, OUTPUT);
    Serial.begin(9600);

    int x = 10;
    peaks.push(&x);
    int *get;
    peaks.pop(get);
    Serial.println(*get, DEC);
    x++;
    peaks.push(&x);
    x++;
    peaks.push(&x);
    x++;
    peaks.push(&x);
    x++;
    peaks.push(&x);
}

/**
 * 心拍数の取得
 * 
 * @return 心拍数
 */
int get_pulse()
{
    int pulse = analogRead(A0);

    return 60;
}

/**
 * 心拍数の取得
 * 
 * @return 心拍数
 */
int get_heart_rate()
{
    // Serial.println(analogRead(A0));

    return 60;
}

/**
 * 点灯時間の計算
 * 
 * @param heart_rate 心拍数
 * @return 点灯時間
 */
unsigned long get_lighting_time(int heart_rate)
{
    // 点灯時間の計算
    float time = (float)60 / (L_COLORS * heart_rate);
    // マイクロ秒変換
    int lighting_time = time * 1000000;

    return lighting_time;
}

/**
 * ディスプレイの描画
 */
void draw_display()
{
    // 描画
    analogWrite(PWM_1, COLORS_PWM_1[i] * 51);
    analogWrite(PWM_2, COLORS_PWM_2[i] * 51);

    // 描画した時刻を保存
    drew_time = micros();

    // Colorsを更新
    if (i == L_COLORS - 1)
    {
        i = 0;
    }
    else
    {
        i++;
    }
}

/**
 * メインループ
 */
void loop()
{
    // 心拍数の取得
    int heart_rate = get_heart_rate();
    // 点灯時間の計算
    unsigned long lighting_time = get_lighting_time(heart_rate);

    // 現在時刻の取得
    unsigned long now = micros();
    // 描画時間の経過で再描画
    if (now > drew_time + lighting_time)
    {
        draw_display();
    }
}

// シリアルスピード
#define SPEED 115200
// 使用ピン
#define PWM_1 5
#define PWM_2 6
// 電圧
#define PWM_HIGH 2
#define PWM_LOW 0
// 閾値
#define THRESHOLD_PULSE 650
#define THRESHOLD_TIME 300000

// 黒の電圧値 (V)
const int BLACK_PWM_1 = PWM_HIGH;
const int BLACK_PWM_2 = PWM_LOW;
// 白の電圧値 (V)
const int WHITE_PWM_1 = PWM_LOW;
const int WHITE_PWM_2 = PWM_HIGH;
// 色データ長
const int L_COLORS = 2;
// 色データ
const int COLORS_PWM_1[L_COLORS] = {WHITE_PWM_1, BLACK_PWM_1};
const int COLORS_PWM_2[L_COLORS] = {WHITE_PWM_2, BLACK_PWM_2};

int heart_rate;              // 心拍数
unsigned long last_peak = 0; // 1つ前のピーク検出時刻
unsigned long drew_time;     // ディスプレイ描画時刻
int i = 0;                   // Colorsカウンタ

/**
 * 初期化
 */
void setup()
{
    pinMode(PWM_1, OUTPUT);
    pinMode(PWM_2, OUTPUT);
    Serial.begin(SPEED);
}

/**
 * 脈波の取得
 */
void get_pulse()
{
    int pulse = analogRead(A0);

    // 心拍数の更新
    update_heart_rate(pulse);
}

/**
 * 心拍数の更新
 * 
 * @param pulse 脈波値
 */
void update_heart_rate(int pulse)
{
    // 現在時刻の取得
    unsigned long now = micros();

    // ピーク検出
    if (now > last_peak + THRESHOLD_TIME && pulse > THRESHOLD_PULSE)
    {
        // ピーク間隔を計算
        unsigned long interval = now - last_peak;
        // 心拍数の更新
        heart_rate = (60 * 1000000) / interval;
        last_peak = now;
    }
}

/**
 * 点灯時間の計算
 * 
 * @return 点灯時間
 */
unsigned long get_lighting_time()
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
    Serial.println(heart_rate);

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
    // 脈波の取得
    get_pulse();

    // 点灯時間の計算
    unsigned long lighting_time = get_lighting_time();

    // 現在時刻の取得
    unsigned long now = micros();
    // 描画時間の経過で再描画
    if (now > drew_time + lighting_time)
    {
        draw_display();
    }
}

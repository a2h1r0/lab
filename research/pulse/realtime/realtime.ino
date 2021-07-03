// シリアルスピード
#define SPEED 115200
// 使用ピン
#define PWM_1 5
#define PWM_2 6
// 電圧
#define PWM_HIGH 2
#define PWM_LOW 0
// 閾値
#define THRESHOLD_PULSE 700
#define THRESHOLD_TIME 500000

// 色データ長
#define L_COLORS 2
#define AVERAGE_SIZE 50

// 黒の電圧値 (V)
const int BLACK_PWM_1 = PWM_HIGH;
const int BLACK_PWM_2 = PWM_LOW;
// 白の電圧値 (V)
const int WHITE_PWM_1 = PWM_LOW;
const int WHITE_PWM_2 = PWM_HIGH;
// 色データ
const int COLORS_PWM_1[L_COLORS] = {WHITE_PWM_1, BLACK_PWM_1};
const int COLORS_PWM_2[L_COLORS] = {WHITE_PWM_2, BLACK_PWM_2};

int heart_rate;              // 心拍数
unsigned long last_peak = 0; // 1つ前のピーク検出時刻
unsigned long drew_time;     // ディスプレイ描画時刻
int i = 0;                   // Colorsカウンタ

int heart_rates[AVERAGE_SIZE]; // 過去心拍数配列（移動平均）
int j = 0;                     // 移動平均カウンタ
const int default_rate = 70;   // 過去心拍数の初期値

/**
 * 初期化
 */
void setup()
{
    pinMode(PWM_1, OUTPUT);
    pinMode(PWM_2, OUTPUT);
    // 初期値は黒
    analogWrite(PWM_1, BLACK_PWM_1 * 51);
    analogWrite(PWM_2, BLACK_PWM_2 * 51);
    delay(10000);

    Serial.begin(SPEED);

    // 過去心拍数配列の初期化
    for (int k = 0; k < AVERAGE_SIZE; k++)
    {
        heart_rates[k] = default_rate;
    }
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
        // 心拍数の計算
        int rate = (60 * 1000000) / interval;
        // 心拍数配列に追加
        heart_rates[j] = rate;

        // 平均値の計算
        int sum = 0;
        for (int k = 0; k < AVERAGE_SIZE; k++)
        {
            sum += heart_rates[k];
        }
        int average = sum / AVERAGE_SIZE;

        // 値の更新
        heart_rate = average;
        last_peak = now;
        if (j == AVERAGE_SIZE - 1)
        {
            j = 0;
        }
        else
        {
            j++;
        }
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
    Serial.println(heart_rate);

    // マイクロ秒変換
    unsigned long lighting_time = time * 1000000;

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

    // Colorsの更新
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

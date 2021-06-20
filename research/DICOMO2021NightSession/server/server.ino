int UP_HIGH = 110;        //UP時のモータ起動時間（仮）
int UP_LOW = 1000;        //UP時のモータ停止時間（仮）
int STAY_HIGH = 100;      //STOP時のモータ起動時間（仮）
int STAY_LOW = 1500;      //STOP時のモータ停止時間（仮）
int DOWN_TIME = 3 * 1000; //降下時間（仮）
int PIN_number = 6;       //リレー１が７番ピン、２が６番、３が５番（×）、４が４番

/**
 * @fn
 * 初期化
 */
void setup()
{
    Serial.begin(9600);
    pinMode(PIN_number, OUTPUT);
}

/**
 * @fn
 * 降下処理
 */
void down()
{
    delay(DOWN_TIME);

    // 処理の終了通知
    Serial.println(0);
}

/**
 * @fn
 * 巻き上げ処理
 */
void up()
{
    for (int i = 0; i < 1000; i++)
    {
        if (i == 3)
        {
            // 処理の終了通知
            Serial.println(0);
        }

        // 巻き上げ開始
        digitalWrite(PIN_number, HIGH);
        delay(UP_HIGH);
        digitalWrite(PIN_number, LOW);
        delay(UP_LOW);
        // 一旦モータ停止
    }
}

/**
 * @fn
 * 停止処理
 */
void stop()
{
    // ぷかぷか浮かすイメージ
    digitalWrite(PIN_number, HIGH);
    delay(STAY_HIGH);
    digitalWrite(PIN_number, LOW);
    delay(STAY_LOW);
}

/**
 * @fn
 * メインループ
 */
void loop()
{
    // Pythonからデータを受信
    if (Serial.available())
    {
        char data = Serial.read();

        switch (data)
        {
        case '0':
            down();
            break;

        case '1':
            up();
            break;

        default:
            break;
        }
    }

    // データを受信するまで静止
    stop();
}

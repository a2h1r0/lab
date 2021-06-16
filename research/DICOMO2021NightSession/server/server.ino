// 巻き上げ時間
const unsigned int UP_TIME = 130;
// 降下時間
const unsigned int DOWN_TIME = 130;

/**
 * @fn
 * 初期化
 */
void setup()
{
    Serial.begin(9600);
}

/**
 * @fn
 * 降下処理
 */
void down()
{
    // なんらかの処理
    delay(DOWN_TIME * 1000);
}

/**
 * @fn
 * 巻き上げ処理
 */
void up()
{
    // なんらかの処理
    delay(UP_TIME * 1000);
}

/**
 * @fn
 * 停止処理
 */
void stop()
{
    // なんらかの処理
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
        int data = Serial.read();

        switch (data)
        {
        case 0:
            down();
            break;

        case 1:
            up();
            break;

        default:
            break;
        }

        // 処理の終了通知
        Serial.println(0);
    }

    // データを受信するまで静止
    stop();
}

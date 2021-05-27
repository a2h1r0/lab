void setup()
{
    Serial.begin(9600);
}

void loop()
{
    // Pythonからデータを受信
    if (Serial.available())
    {
        String data = Serial.readStringUntil('\0');

        // なんらかの処理

        // 処理の終了通知
        Serial.println(0);
    }
}

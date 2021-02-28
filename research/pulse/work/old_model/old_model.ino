#include "Arduino.h"

/**
 * @fn
 * 初期化
 */
void setup()
{
    Serial.begin(115200);
}

/**
 * @fn
 * メインループ
 */
void loop()
{
    Serial.print(micros());
    Serial.print(",");
    Serial.println(analogRead(A1));
}

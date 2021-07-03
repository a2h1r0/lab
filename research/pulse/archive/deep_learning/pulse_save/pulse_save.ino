#include "Arduino.h"

/**
 * @fn
 * 初期化
 */
void setup()
{
    Serial.begin(14400);
}

/**
 * @fn
 * メインループ
 */
void loop()
{
    Serial.print(micros());
    Serial.print(",");
    Serial.println(analogRead(A0));
}

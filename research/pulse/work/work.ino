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
    Serial.println(analogRead(A1));
}

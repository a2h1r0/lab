#include "Arduino.h"

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
 * メインループ
 */
void loop()
{
    Serial.print(analogRead(A0));
    Serial.print(",");
    Serial.println(analogRead(A1));
}

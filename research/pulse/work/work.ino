#include <Arduino.h>

void setup()
{
  Serial.begin(115200);
}

void loop()
{
  Serial.print(micros());
  Serial.print(",");
  Serial.println(analogRead(A0));
  // ここでPCから信号が流れるまで待機する．1つの値のみを送信．
  delay(10);
}

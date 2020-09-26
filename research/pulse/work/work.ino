#include <Arduino.h>

void setup() {
  Serial.begin(115200);
}

void loop() {
//  Serial.print(micros());
//  Serial.print(",");
  Serial.println(analogRead(A0));
  delay(10);
}

#include <Arduino.h>

//int x=200;
//int y=1000;
void setup() {
Serial.begin(115200);

}

void loop() {
//Serial.print(micros());
//Serial.print(",");
Serial.println(analogRead(A0));
//Serial.print(",");
//Serial.println(analogRead(A1));
delay(10);
}

//Serial.print(x);
//Serial.print(",");
//Serial.print(y);
//Serial.print(",");

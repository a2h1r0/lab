void setup()
{
    Serial.begin(14400);
}

void loop()
{
    Serial.print(micros());
    Serial.print(",");
    Serial.println(analogRead(A0));
}

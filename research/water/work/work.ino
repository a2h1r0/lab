const int RELAY_PIN = 7;

void setup()
{
    pinMode(RELAY_PIN, OUTPUT);
    Serial.begin(9600);
}

void loop()
{
    digitalWrite(RELAY_PIN, HIGH);
    delay(3000);
    digitalWrite(RELAY_PIN, LOW);
    delay(3000);
}

#include <Wire.h>
#include "Adafruit_MPRLS.h"

#define RESET_PIN  -1  // set to any GPIO pin # to hard-reset on begin()
#define EOC_PIN    -1  // set to any GPIO pin to read end-of-conversion by pin
Adafruit_MPRLS mpr = Adafruit_MPRLS(RESET_PIN, EOC_PIN);
#define LED 13 // LED pin


void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  delay(1000);
  Serial.println("MPRLS Simple Test");
  Wire.begin();
  if (! mpr.begin()) {
    Serial.println("Failed to communicate with MPRLS sensor, check wiring?");
    while (1) {
      delay(10);` 
    }
  }
  Serial.println("Found MPRLS sensor");
  pinMode(LED, OUTPUT);

}

void loop() {
  // put your main code here, to run repeatedly:
  float pressure_hPa = mpr.readPressure();
  // Serial.print("Pressure (Pa): "); 
  Serial.println(pressure_hPa*100);
  digitalWrite(LED,HIGH);
  // delay(500);
  // Serial.print("Pressure (PSI): "); Serial.println(pressure_hPa / 68.947572932);
  digitalWrite(LED,LOW);
  delay(10);
}

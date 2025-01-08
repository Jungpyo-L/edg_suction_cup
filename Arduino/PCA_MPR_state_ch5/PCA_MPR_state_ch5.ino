#include "Wire.h"
//#include "Adafruit_MPRLS.h"
#include "Adafruit_MPRLS_JP.h"

#define PCAADDR 0x70

int led = LED_BUILTIN;

// Define states
#define IDLE 1 //this is 3 in the ROS
#define STREAMING 2 //this is 2 in the ROS
int state = IDLE; // initial state

#define RESET_PIN  -1  // set to any GPIO pin # to hard-reset on begin()
#define EOC_PIN    -1  // set to any GPIO pin to read end-of-conversion by pin
Adafruit_MPRLS mpr0 = Adafruit_MPRLS(RESET_PIN, EOC_PIN);
Adafruit_MPRLS mpr1 = Adafruit_MPRLS(RESET_PIN, EOC_PIN);
Adafruit_MPRLS mpr2 = Adafruit_MPRLS(RESET_PIN, EOC_PIN);
Adafruit_MPRLS mpr3 = Adafruit_MPRLS(RESET_PIN, EOC_PIN);
Adafruit_MPRLS mpr4 = Adafruit_MPRLS(RESET_PIN, EOC_PIN);

void pcaselect(uint8_t i) {
  if (i > 5) return;
 
  Wire.beginTransmission(PCAADDR);
  Wire.write(1 << i);
  Wire.endTransmission();  
}


// standard Arduino setup()
void setup()
{
  Serial.begin(115200);
  Serial.println("\n Start setup");
  // while (!Serial);
  Wire.begin();
  Wire.setClock(400000);
  delay(10);
  
  // for (uint8_t t=0; t<4; t++) {
  //     pcaselect(t);
  //     Serial.print("PCA Port #"); Serial.println(t);

  //     for (uint8_t addr = 0; addr<=127; addr++) {
  //       if (addr == PCAADDR) continue;

  //       Wire.beginTransmission(addr);
  //       if (!Wire.endTransmission()) {
  //         Serial.print("Found I2C 0x");  Serial.println(addr,HEX);
  //       }
  //     }
  //   }
  //   Serial.println("\ndone");
    
  Serial.println("\nPCAScanner ready!");

  pcaselect(0);
  if (! mpr0.begin()) {
    Serial.println("Failed to communicate with MPRLS sensor0, check wiring?");
    while (1) {
      delay(10);
    }
  }
  Serial.println("Found MPRLS sensor 0");

  pcaselect(1);
  if (! mpr1.begin()) {
    Serial.println("Failed to communicate with MPRLS sensor1, check wiring?");
    while (1) {
      delay(10);
    }
  }
  Serial.println("Found MPRLS sensor 1");

  
  pcaselect(2);
  if (! mpr2.begin()) {
    Serial.println("Failed to communicate with MPRLS sensor2, check wiring?");
    while (1) {
      delay(10);
    }
  }
  Serial.println("Found MPRLS sensor 2");

  pcaselect(3);
  if (! mpr3.begin()) {
    Serial.println("Failed to communicate with MPRLS sensor3, check wiring?");
    while (1) {
      delay(10);
    }
  }
  Serial.println("Found MPRLS sensor 3");

  pcaselect(4);
  if (! mpr4.begin()) {
    Serial.println("Failed to communicate with MPRLS sensor4, check wiring?");
    while (1) {
      delay(10);
    }
  }
  Serial.println("Found MPRLS sensor 4");

  pinMode(led, OUTPUT);
}

void loop() 
{
  switch (state)
  {
    case IDLE:
    {
      digitalWrite(led, HIGH);
      delay(100);
      digitalWrite(led, LOW);
      delay(100);
      byte incoming = Serial.read();
      if (incoming == 's')
      {
        state = STREAMING;
        digitalWrite(led,HIGH);
      }
      break;
    }
    case STREAMING:
    {
      byte incoming = Serial.read();
      if (incoming == 'i')
      {
        state = IDLE;
      }
      for (int i = 0; i <= 4; i++) {
        pcaselect(i);
        if (i == 0) {
          mpr0.requestData();
        }else if (i == 1) {
          mpr1.requestData();
        }else if (i == 2) {
          mpr2.requestData();
        }else if (i == 3) {
          mpr3.requestData();
        }else if (i == 4) {
          mpr4.requestData();
        }
      }

      pcaselect(0);
      float pressure_Pa0 = mpr0.readPressure2()*100;
      pcaselect(1);
      float pressure_Pa1 = mpr1.readPressure2()*100;
      pcaselect(2);
      float pressure_Pa2 = mpr2.readPressure2()*100;
      pcaselect(3);
      float pressure_Pa3 = mpr3.readPressure2()*100;
      pcaselect(4);
      float pressure_Pa4 = mpr4.readPressure2()*100;
      Serial.print(pressure_Pa0);
      Serial.print(" ");
      Serial.print(pressure_Pa1);
      Serial.print(" ");
      Serial.print(pressure_Pa2);
      Serial.print(" ");
      Serial.print(pressure_Pa3);
      Serial.print(" ");
      Serial.println(pressure_Pa4);
      break;
    }
  }
 
}

#define PWM_pin 26  // A0 in ESP32
#define Push_pin 25 // A1
#define Pull_pin 4 // A5
#define Pull_pin_below 15 //D15
#define PULL_state 1
#define PUSH_state 2
#define STOP 0

const int freq = 30;
const int pwmChannel = 1;
const int resolution = 8;

int state = 0; // Variable to store state
int pwm = 0;   // Variable to store PWM

void setup() {
  Serial.begin(115200);
  Serial.println("ESP32 PWM Controller Ready");

  // Setup PWM channel
  ledcSetup(pwmChannel, freq, resolution);
  ledcAttachPin(PWM_pin, pwmChannel);

  // Setup push and pull pins
  pinMode(Push_pin, OUTPUT);
  pinMode(Pull_pin, OUTPUT);
  pinMode(Pull_pin_below, OUTPUT);
}

void loop() {
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n'); // read the whole line once
    input.trim();                                // remove spaces and any '\r'

    int underscore = input.indexOf('_');

    if (underscore >= 0) {
      // format: "<state>_<pwm>"
      String s_state = input.substring(0, underscore);
      String s_pwm   = input.substring(underscore + 1);
      state = s_state.toInt();
      pwm   = s_pwm.toInt();
    } else {
      // format: "<pwm>" only
      pwm = input.toInt();
      if (pwm == 0) {
        state = STOP;   // 0
      } else {
        state = PULL_state;      // Active
      }
    }
    // keep PWM in range 0–100
    pwm = constrain(pwm, 0, 100);
  }

  // Map the PWM value (assume input range is 0-100) to 8-bit range (0-255)
  int pwm_8bit = map(pwm, 0, 100, 0, 255);

  switch (state)
  {
  case STOP:
    ledcWrite(pwmChannel, 0);
    digitalWrite(Push_pin, LOW);
    digitalWrite(Pull_pin, LOW);
    digitalWrite(Pull_pin_below, LOW);
    break;
  case PUSH_state:
    ledcWrite(pwmChannel, pwm_8bit);
    digitalWrite(Push_pin, HIGH);
    digitalWrite(Pull_pin, LOW);
    digitalWrite(Pull_pin_below, LOW);
    break;
  
  case PULL_state:
    ledcWrite(pwmChannel, pwm_8bit);
    digitalWrite(Push_pin, LOW);
    digitalWrite(Pull_pin, HIGH);
    digitalWrite(Pull_pin_below, HIGH);
    break;
  }
}

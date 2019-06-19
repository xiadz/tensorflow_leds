// A simple tool to light up a strip or matrix of LEDs according
// to what's sent on the serial input.
//
// Protocol:
//   - \xff as the start byte
//   - Then raw byte values for R, G, B on LED0
//     - Then raw byte values for R, G, B on LED1
//     - and so on.
//   - Finally, a checksum byte, which is a uint8_t sum
//     of all the color values.

// Outputs (on the serial port) number of successful
// and unsuccessful reads and other stats.
// It also flips the LED_BUILTIN on
// on a failed receive (cleared soon after; errors
// result in chaotic blinking).

// Do not use 255 ('\xff') for LED color values,
// this code will reject such transmission.
// This is a simple hack to make it easier to
// not start processing in the middle of the stream.

// The program applies gamma correction to the LED color
// values.


#include <FastLED.h>

//
// Constants.
//

// There's 16*16 LEDs in the matrix.
#define NUM_LEDS (16*16)

// LEDs are at pin 8.
#define DATA_PIN 8

// Speed of the serial port.
#define SERIAL_BAUD 230400

// Whether to enable gamma correction.
#define ENABLE_GAMMA_CORRECTION true

// A division constant for all RGB values.
// Applied after gamma correction.
// This is here mostly to not burn the LED
// strip/matrix with excessive power.
#define DIVIDE_ALL_COLORS 2

// How often should status updates be printed.
#define STATUS_UPDATE_MS (30 * 1000)

//
// LEDs.
//

// Keeps all RGB values to be written to LEDs.
CRGB leds[NUM_LEDS];

//
// Runtime statistics.
//

// How many receives were successful.
uint32_t total_successful_rcvs = 0;

// How many receives were failed.
uint32_t total_failed_rcvs = 0;

// Time (in ms) of last successful serial port read.
unsigned long last_serial_read_ms = 0;

// Time (in ms) of last successful FastLED show().
unsigned long last_show_ms = 0;


// Turns off all LEDs.
void turn_off_leds() {
  for (int i = 0; i < NUM_LEDS; ++i) {
    leds[i] = CRGB(0, 0, 0);
  }
  FastLED.show();
}

// Arduino setup function.
void setup() {
  Serial.begin(SERIAL_BAUD);
  while (!Serial) { }

  // Set up the built in LED.
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);

  // Set up FastLED.
  FastLED.addLeds<NEOPIXEL, DATA_PIN>(leds, NUM_LEDS);

  // Turn off all LEDs.
  // Those may be still on from pre-boot time.
  turn_off_leds();
}

// Gamma correction table for the LEDs.
const uint8_t PROGMEM gamma8[] = {
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,
    1,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,
    2,  3,  3,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4,  5,  5,  5,
    5,  6,  6,  6,  6,  7,  7,  7,  7,  8,  8,  8,  9,  9,  9, 10,
   10, 10, 11, 11, 11, 12, 12, 13, 13, 13, 14, 14, 15, 15, 16, 16,
   17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 24, 24, 25,
   25, 26, 27, 27, 28, 29, 29, 30, 31, 32, 32, 33, 34, 35, 35, 36,
   37, 38, 39, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 50,
   51, 52, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 66, 67, 68,
   69, 70, 72, 73, 74, 75, 77, 78, 79, 81, 82, 83, 85, 86, 87, 89,
   90, 92, 93, 95, 96, 98, 99,101,102,104,105,107,109,110,112,114,
  115,117,119,120,122,124,126,127,129,131,133,135,137,138,140,142,
  144,146,148,150,152,154,156,158,160,162,164,167,169,171,173,175,
  177,180,182,184,186,189,191,193,196,198,200,203,205,208,210,213,
  215,218,220,223,225,228,231,233,236,239,241,244,247,249,252,255 };

// Gamma corrects a single value.
uint8_t gamma_correct(uint8_t input) {
  if (ENABLE_GAMMA_CORRECTION) {
    return pgm_read_byte(&gamma8[input]);
  }
  return input;
}

// Blocks until a new character is available
// on the serial port, and returns that character.
uint8_t blocking_read_char() {
  while(Serial.available() == 0) { }
  return Serial.read();
}

// Read LEDs setup from the serial port and write
// the configuration to the LED matrix/strip.
//
// Will abort if byte 255 is found in the middle
// of the transmission, or if the checksum does
// not match.
//
// Returns true if data was received correctly and
// pushed to the LEDs, and false otherwise.
bool update_leds() {
  const unsigned long read_start_millis = millis();
  uint8_t computed_checksum = 0;
  for (int i = 0; i < NUM_LEDS; ++i) {
    const uint8_t r = blocking_read_char();
    computed_checksum += r;
    const uint8_t g = blocking_read_char();
    computed_checksum += g;
    const uint8_t b = blocking_read_char();
    computed_checksum += b;
    if (r == 255 || g == 255 || b == 255) {
      // Illegal byte.
      return false;
    }
    leds[i] = CRGB(
      gamma_correct(r) / DIVIDE_ALL_COLORS,
      gamma_correct(g) / DIVIDE_ALL_COLORS,
      gamma_correct(b) / DIVIDE_ALL_COLORS);
  }
  const uint8_t checksum = blocking_read_char();
  if (checksum != computed_checksum) {
    // Bad checksum.
    return false;
  }

  // Read was successful.
  last_serial_read_ms = millis() - read_start_millis;

  const unsigned long show_start_millis = millis();
  FastLED.show();
  last_show_ms = millis() - show_start_millis;

  return true;
}

// Attempts to receive a LED colors transmission.
// Will return immediately if there's no data
// pending to be read.
void try_receive() {
  const int character = Serial.read();
  if (character != 255) {
    return;
  }

  // Clear the error LED.
  digitalWrite(LED_BUILTIN, LOW);

  // New transmission
  if (update_leds()) {
    total_successful_rcvs++;
  } else {
    total_failed_rcvs++;
    // Enable the error LED.
    digitalWrite(LED_BUILTIN, HIGH);
  }
}

// Prints a status update if enough time has passed.
void maybe_status_update() {
  // Time of last serial port status update.
  static unsigned long last_status_millis = 0;

  if (last_status_millis == 0) {
    // First entrance.
    last_status_millis = millis();
    return;
  }

  if (millis() - last_status_millis < STATUS_UPDATE_MS) {
    return;
  }


  // Last value of receives.
  // Used to compute FPS.
  static uint32_t last_successful_rcvs = 0;
  static uint32_t last_failed_rcvs = 0;

  // Compute FPS.
  const unsigned long time_since_last = millis() - last_status_millis;
  const float successful_fps =
    (float)(total_successful_rcvs - last_successful_rcvs) /
    (float)(time_since_last) * 1000.0f;
  const float failed_fps =
    (float)(total_failed_rcvs - last_failed_rcvs) /
    (float)(time_since_last) * 1000.0f;

  Serial.print("Successful receives FPS: ");
  Serial.println(successful_fps);
  Serial.print("Failed receives FPS: ");
  Serial.println(failed_fps);

  Serial.print("Total successful receives: ");
  Serial.println(total_successful_rcvs);
  Serial.print("Total failed receives: ");
  Serial.println(total_failed_rcvs);

  Serial.print("Last serial read latency [ms]: ");
  Serial.println(last_serial_read_ms);
  Serial.print("Last LED show latency [ms]: ");
  Serial.println(last_show_ms);

  Serial.print("millis:");
  Serial.println(millis());

  last_status_millis = millis();
  last_successful_rcvs = total_successful_rcvs;
  last_failed_rcvs = total_failed_rcvs;
}

// Arduino loop.
void loop() {
  try_receive();
  maybe_status_update();
}

// A simple tool to light up LEDs according to what's sent
// on the input.
//
// Protocol:
//   \xff as the start byte
//   Then raw byte values for R, G, B on LED0
//   Then raw byte values for R, G, B on LED1
//   and so on.

// Please do not use 255 ('\xff') for LED color values,
// so that this code does not start processing in the middle
// of the stream.

#include <FastLED.h>

// There's 60 LEDs on the strip.
#define NUM_LEDS 30
#define DATA_PIN 8

// Speed of the serial port
#define SERIAL_BAUD 115200

CRGB leds[NUM_LEDS];

void setup() {
  Serial.begin(SERIAL_BAUD);
  while (!Serial) { }

  FastLED.addLeds<NEOPIXEL, DATA_PIN>(leds, NUM_LEDS);
}

uint8_t blocking_read_char() {
  int character;
  do {
    character = Serial.read();
  } while(character == -1);
  return character;
}

void update_leds() {
  for (int i = 0; i < NUM_LEDS; ++i) {
    const uint8_t r = blocking_read_char();
    const uint8_t g = blocking_read_char();
    const uint8_t b = blocking_read_char();
    leds[i] = CRGB(r, g, b);
  }
  FastLED.show();
}

void loop() {
  int character = blocking_read_char();
  if (character == 255) {
    // New transmission
    update_leds();
  }
}

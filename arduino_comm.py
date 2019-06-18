import numpy as np
import termios
import threading

ARDUINO_DEVICE="/dev/cu.usbserial-1420"
# With 16*16*3 bytes to transfer each message is 771
# bytes long, and therefore should take around 26.7 msec
# to transfer over the serial port (at 230.4k baud).
ARDUINO_COMM_SPEED=230400
NUM_LEDS_HEIGHT=16
NUM_LEDS_WIDTH=16
NUM_LEDS_COLORS=3


class ArduinoDevice(object):

    def __init__(self):
        """Opens Arduino device controlling the LEDs."""
        print("Opening Arduino at", ARDUINO_DEVICE,
              "baud:", ARDUINO_COMM_SPEED)
        self.device = open(ARDUINO_DEVICE, "wb")
        self._set_speeds(self.device)
        self.device_status = open(ARDUINO_DEVICE, "r")
        self._set_speeds(self.device_status)

        self.status_reader_thread = threading.Thread(
            target=self._status_reader_loop)
        self.status_reader_thread.setDaemon(True)
        self.status_reader_thread.start()

    def _set_speeds(self, handler):
        flags = termios.tcgetattr(handler.fileno())
        flags[4] = ARDUINO_COMM_SPEED
        flags[5] = ARDUINO_COMM_SPEED
        termios.tcsetattr(handler.fileno(), termios.TCSANOW, flags)

    def _status_reader_loop(self):
        while True:
            line = self.device_status.readline()
            line = line.strip()
            if line:
                print("Arduino status:", line)

    def send_to_device(self, raw_data):
        """Sends a single full LED update to Arduino.

        The data must come as a Numpy np.uint8 array with
        the following dimensions:

        (NUM_LEDS_HEIGHT, NUM_LEDS_WIDTH, NUM_LEDS_COLORS)

        Colors must come as R, G, B.
        """
        if raw_data.dtype != np.uint8:
            raise ValueError("Passed data is not in the np.uint8 dtype")
        expected_shape = (NUM_LEDS_HEIGHT, NUM_LEDS_WIDTH, NUM_LEDS_COLORS)
        if raw_data.shape != expected_shape:
            raise ValueError("Passed data has invalid shape")

        # Start composing the message.
        message = []

        # The initiate transmission byte.
        message.append(255)

        # Will hold the checksum value.
        checksum = 0

        # Attach all LED color values.
        for y in range(NUM_LEDS_HEIGHT):
            for x in range(NUM_LEDS_WIDTH):
                for c in range(NUM_LEDS_COLORS):
                    # On even y's the x's are inverted.
                    if y % 2 == 0:
                        x_flipped = 15 - x
                    else:
                        x_flipped = x
                    value = raw_data[y, x_flipped, c]
                    # Capped at 254, not 255, to avoid sending the '\xff' byte
                    # in the middle of transmission.
                    value = max(min(int(value), 254), 0)
                    message.append(value)
                    checksum += value

        # Attach checksum.
        checksum %= 256
        message.append(checksum)

        # Send the message.
        message = bytearray(message)
        self.device.write(message)
        self.device.flush()

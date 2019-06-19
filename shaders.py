#!/usr/bin/env python3

import colorsys
import math
import numpy as np
import time

import arduino_comm
import fps_tools

device = arduino_comm.ArduinoDevice()
fps_meter = fps_tools.FPSMeter()
fps_limiter = fps_tools.FPSLimiter()

def compute_shader_xy(t, x, y):
    r = math.sqrt(x**2.0 + y**2.0)
    hue = 0.0
    hue += t * -0.5
    hue += r * 2.0
    hue %= 1.0

    saturation = 1.0

    value = (math.sin(5.0 * r - 5.0 * t) + 1.0) * 0.5
    value *=  0.5

    return colorsys.hsv_to_rgb(hue, saturation, value)

def compute_shader(time_sec):
    frame = np.zeros((16, 16, 3))
    dx = 2.0 / 15.0
    for y in range(16):
        for x in range(16):
            point_x = (x - 7.5) * dx
            point_y = (y - 7.5) * dx
            frame[y, x] = compute_shader_xy(time_sec, point_x, point_y)
    return frame

def floats_to_uints(frame_floats):
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    for y in range(16):
        for x in range(16):
            for c in range(3):
                val = frame_floats[y, x, c]
                val = min(max(int(val * 255.0), 0), 255)
                frame[y, x, c] = val
    return frame

while True:
    time_sec = time.monotonic()
    frame_floats = compute_shader(time_sec)
    frame = floats_to_uints(frame_floats)
    device.send_to_device(frame)

    fps_limiter.frame_completed()
    fps_meter.frame_completed()

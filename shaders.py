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

def flower_shader(t, x, y):
    r = math.sqrt(x**2.0 + y**2.0)
    theta = math.atan2(y, x)

    hue = 0.0
    hue += 1.0 * r
    hue %= 1.0

    saturation = 1.0

    if r < 0.25:
        value = 1.0
    elif r > 1.2:
        value = 0.0
    else:
        flower_speed = 10.0
        flower_leaves = 5
        value_angle = 0.0
        value_angle += (flower_speed / 2.0) * r
        value_angle += -flower_speed * t
        value_angle += flower_leaves * theta
        value = (math.sin(value_angle) + 1.0) * 0.5

    value *=  0.5

    return colorsys.hsv_to_rgb(hue, saturation, value)

def compute_shader(time_sec, shader):
    frame = np.zeros((16, 16, 3))
    dx = 2.0 / 15.0
    for y in range(16):
        for x in range(16):
            point_x = (x - 7.5) * dx
            point_y = (y - 7.5) * dx
            frame[y, x] = shader(time_sec, point_x, point_y)
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
    frame_floats = compute_shader(time_sec, flower_shader)
    frame = floats_to_uints(frame_floats)
    device.send_to_device(frame)

    fps_limiter.frame_completed()
    fps_meter.frame_completed()

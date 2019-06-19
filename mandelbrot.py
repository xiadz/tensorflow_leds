#!/usr/bin/env python3

import colorsys
import numpy as np
import time

import arduino_comm
import fps_meter

device = arduino_comm.ArduinoDevice()
fps_meter = fps_meter.FPSMeter()

center = -0.7453 + 0.1127j
zoom = 20.0
zoom_per_frame = 0.95
max_iter = 1000

def mandelbrot_color(point):
    x = 0.0
    for i in range(max_iter):
        x = x*x + point
        if abs(x) > 2.0:
            break
    if i == max_iter - 1:
        return [0, 0, 0]
    r, g, b = colorsys.hsv_to_rgb((i / 30.0) % 1.0, 1.0, 0.3)
    return int(r * 255.0), int(g * 255.0), int(b * 255.0) 


def compute_mandelbrot(center, zoom):
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    dx = zoom / 15
    for y in range(16):
        for x in range(16):
            point = center
            point += (x - 7.5) * dx
            point += (y - 7.5) * dx * 1j
            r, g, b = mandelbrot_color(point)
            frame[y, x, 0] = r
            frame[y, x, 1] = g
            frame[y, x, 2] = b
            
    return frame

while True:
    zoom *= zoom_per_frame
    frame = compute_mandelbrot(center, zoom)
    device.send_to_device(frame)
    fps_meter.frame_completed()

    time.sleep(0.1)

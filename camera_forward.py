#!/usr/bin/env python3

import numpy as np

import arduino_comm
import camera_access
import fps_tools

device = arduino_comm.ArduinoDevice()
cap = camera_access.open_camera()
fps_meter = fps_tools.FPSMeter()
fps_limiter = fps_tools.FPSLimiter()

print("Running ...")
while True:
    frame = camera_access.get_camera_frame(cap)
    frame = camera_access.square_frame(frame)
    frame = camera_access.scale_frame(frame, 16, 16)
    device.send_to_device(frame)

    fps_limiter.frame_completed()
    fps_meter.frame_completed()

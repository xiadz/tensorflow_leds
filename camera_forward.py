#!/usr/bin/env python3

import cv2
import numpy as np
import time

import arduino_comm
import camera_access
import fps_meter


device = arduino_comm.ArduinoDevice()
cap = camera_access.open_camera()
fps_meter = fps_meter.FPSMeter()

print("Running ...")
while True:
    frame = camera_access.get_camera_frame(cap)
    frame = camera_access.square_frame(frame)
    frame = camera_access.scale_frame(frame, 16, 16)
    device.send_to_device(frame)
    fps_meter.frame_completed()

    time.sleep(0.1)

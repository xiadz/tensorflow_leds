#!/usr/bin/env python3

import os
os.environ["TFHUB_CACHE_DIR"] = "/Users/mosowski/prog/led_features/tfcache"

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import time

ARDUINO_DEVICE="/dev/cu.usbmodem14201"
TF_MODULE="https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/3"
NUM_LEDS=30

def value_transform(value):
    value -= 0.5
    value *= 0.3 * 255.0
    value = int(value)
    # Capped at 254, not 255, to avoid sending the '\xff' byte
    # in the middle of transmission.
    value = max(min(value, 254), 0)
    return value

def send_to_device(device, results):
    message = []
    message.append(255)
    for i in range(NUM_LEDS * 3):
        value = results[i]
        message.append(value_transform(value))

    message = bytearray(message)
    device.write(message)
    device.flush()

def prepare_frame_batch(frame, module):
    frame_height, frame_width, frame_colors = frame.shape
    module_input_height, module_input_width = hub.get_expected_image_size(module)

    # Crop and scale to be an input for the module.
    frame_square_size = min(frame_height, frame_width)
    frame_y_offset = (frame_height - frame_square_size) // 2
    frame_x_offset = (frame_width - frame_square_size) // 2
    frame_cropped = frame[
        frame_y_offset:frame_y_offset + frame_square_size,
        frame_x_offset:frame_x_offset + frame_square_size]
    frame_scaled = cv2.resize(
        frame_cropped,
        (module_input_height, module_input_width),
        interpolation=cv2.INTER_CUBIC)

    # Tensorflow expects values between 0.0 and 1.0, and we have
    # between 0.0 and 255.0
    frame_scaled = frame_scaled / 255.0

    # Tensorflow needs the data as a batch of size 1.
    frame_batch = np.expand_dims(frame_scaled, 0)

    return frame_batch

def single_iteration(device, cap, sess, results_output, frame_placeholder, module):
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Couldn't get the image")

    frame_batch = prepare_frame_batch(frame, module)

    # Run Tensorflow.
    results = sess.run(results_output, feed_dict={frame_placeholder: frame_batch})

    # Send data to the device.
    send_to_device(device, results)

    # Print data for the user.
    for i in range(10):
        print("%.2f" % results[i], end = ' ')
    print()

def start_tf(module, cap, device):
    print("Preparing TensorFlow")
    module_input_height, module_input_width = hub.get_expected_image_size(module)
    frame_placeholder = tf.compat.v1.placeholder(
        dtype=tf.float32,
        shape=(1, module_input_height, module_input_width, 3))
    results_output = tf.squeeze(module(frame_placeholder))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("Running loop")
        while True:
            single_iteration(
                device, cap, sess, results_output,
                frame_placeholder, module)

def open_port(module, cap):
    with open(ARDUINO_DEVICE, "wb") as device:
        start_tf(module, cap, device)

def open_video(module):
    print("Initializing video input")
    cap = cv2.VideoCapture(0)
    try:
        open_port(module, cap)
    finally:
        cap.release()

print("Loading TensorFlow hub module %s" % TF_MODULE)
module = hub.Module(TF_MODULE)
open_video(module)

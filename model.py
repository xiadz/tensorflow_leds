#!/usr/bin/env python3

import os
os.environ["TFHUB_CACHE_DIR"] = "./tfcache"

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import time

import arduino_comm
import camera_access
import nn_data_packager

TF_MODULE="https://tfhub.dev/google/imagenet/mobilenet_v2_100_96/feature_vector/3"
FPS_LIMIT=10.0
NN_OUTPUT_ORDER_FILE="reordering_outputs/nn_outputs_order.csv"

print("Reading NN outputs order from %s" % NN_OUTPUT_ORDER_FILE)
nn_outputs_order = np.genfromtxt(NN_OUTPUT_ORDER_FILE, dtype=int)


def single_iteration(device, cap, sess, results_output, frame_placeholder, module):
    module_input_height, module_input_width = hub.get_expected_image_size(module)

    frame = camera_access.get_camera_frame(cap)
    frame_squared = camera_access.square_frame(frame)
    frame_scaled = camera_access.scale_frame(
        frame, module_input_height, module_input_width)

    # Tensorflow expects values between 0.0 and 1.0, and we have
    # between 0.0 and 255.0
    frame_scaled = frame_scaled / 255.0

    # Tensorflow needs the data as a batch of size 1.
    frame_batch = np.expand_dims(frame_scaled, 0)

    # Run Tensorflow.
    results = sess.run(results_output, feed_dict={frame_placeholder: frame_batch})

    # Package data.
    results = nn_data_packager.package_data(results, nn_outputs_order)

    # Send data to the device.
    device.send_to_device(results)

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
        frame_number = 0
        before_for_fps = time.monotonic()
        while True:
            before = time.monotonic()
            single_iteration(
                device, cap, sess, results_output,
                frame_placeholder, module)
            frame_number += 1
            if frame_number % 100 == 0:
                after_for_fps = time.monotonic()
                print("FPS: %.1f" % (100.0 / (after_for_fps - before_for_fps)))
                before_for_fps = time.monotonic()
            after = time.monotonic()
            elapsed_time = after - before
            min_time = 1.0 / FPS_LIMIT
            if min_time > elapsed_time:
                time.sleep(min_time - elapsed_time)


print("Loading TensorFlow hub module %s" % TF_MODULE)
module = hub.Module(TF_MODULE)
cap = camera_access.open_camera()
device = arduino_comm.ArduinoDevice()
start_tf(module, cap, device)

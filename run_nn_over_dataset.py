#!/usr/bin/env python3

import os
os.environ["TFHUB_CACHE_DIR"] = "./tfcache"

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import time

TF_MODULE="https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/3"
DATASET="cifar100"
OUTPUT_FILE="raw_nn_outputs.csv"


print("Loading TensorFlow hub module %s" % TF_MODULE)
module = hub.Module(TF_MODULE)

print("Loading TensorFlow dataset %s" % DATASET)
# Only TEST split, that's 10k images and should be enough.
dataset = tfds.load(name=DATASET, split=tfds.Split.TEST)
dataset = dataset.batch(32)

module_input_height, module_input_width = hub.get_expected_image_size(module)
frame_placeholder = tf.compat.v1.placeholder(
    dtype=tf.float32,
    shape=(None, module_input_height, module_input_width, 3))
results_output = module(frame_placeholder)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    iterator =  dataset.make_one_shot_iterator()
    images_slice = iterator.get_next()["image"]
    # Network expects floats between 0.0 and 1.0.
    images_slice = tf.dtypes.cast(images_slice, dtype=tf.float32) / 255.0
    # Need to rescale to input size.
    images_slice = tf.image.resize_images(images_slice, (module_input_height, module_input_width))
    # Run the module.
    results = module(images_slice)

    print("Running loop")
    with open("raw_nn_outputs.csv", "w") as f:
        while True:
            data = sess.run(results)

            # Write to the CSV.
            for row in data:
                f.write(", ".join(map(str, row)))
                f.write("\n")


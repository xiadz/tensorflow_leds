import cv2
import numpy as np


def open_camera():
    print("Opening camera video input")
    cap = cv2.VideoCapture(0)
    return cap


def get_camera_frame(cap):
    """Returns a single camera frame in RGB."""
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Couldn't get image from the camera")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


def square_frame(frame):
    """Squares a frame by discarding pixels."""
    frame_height, frame_width, frame_colors = frame.shape

    frame_square_size = min(frame_height, frame_width)
    frame_y_offset = (frame_height - frame_square_size) // 2
    frame_x_offset = (frame_width - frame_square_size) // 2
    frame_cropped = frame[
        frame_y_offset:frame_y_offset + frame_square_size,
        frame_x_offset:frame_x_offset + frame_square_size]

    return frame_cropped


def scale_frame(frame, height, width):
    frame_scaled = cv2.resize(
        frame, (height, width), interpolation=cv2.INTER_CUBIC)

    return frame_scaled

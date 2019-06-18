import time

class FPSMeter(object):

    def __init__(self):
        self._before_for_fps = time.monotonic()
        self._frame_number = 0
        self._update_every = 100

    def frame_completed(self):
        self._frame_number += 1
        if self._frame_number % self._update_every == 0:
            time_took = (time.monotonic() - self._before_for_fps)
            fps = float(self._update_every) / time_took
            print("FPS: %.1f" % fps)
            self._before_for_fps = time.monotonic()

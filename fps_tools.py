import time

class FPSMeter(object):
    """A class to measure FPS. Prints to stdout."""

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

class FPSLimiter(object):
    """A class to limit FPS."""
    def __init__(self):
        self._before = time.monotonic()
        self._fps_limit = 15.0
        self._min_time = 1.0 / self._fps_limit

    def frame_completed(self):
        """Sleeps for a while if program is going too fast."""
        after = time.monotonic()
        elapsed = after - self._before
        too_fast_by = self._min_time - elapsed
        if too_fast_by > 0.0:
            time.sleep(too_fast_by)
        self._before = time.monotonic()

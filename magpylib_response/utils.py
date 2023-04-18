import threading
import time
from contextlib import contextmanager

from loguru import logger


class ElapsedTimeThread(threading.Thread):
    """ "Stoppable thread that logs the time elapsed"""

    def __init__(self, msg=None, min_log_time=1):
        super().__init__()
        self._stop_event = threading.Event()
        self.thread_start = time.time()
        self.msg = msg
        self.min_log_time = min_log_time
        self._msg_displayed = False

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

    def getStart(self):
        return self.thread_start

    def run(self):
        self.thread_start = time.time()
        while not self.stopped():
            if (
                self.msg is not None
                and time.time() - self.thread_start > self.min_log_time
                and not self._msg_displayed
            ):
                logger.opt(colors=True).info(f"Start {self.msg}")
                self._msg_displayed = True
            # include a delay here so the thread doesn't uselessly thrash the CPU
            time.sleep(max(0.01, self.min_log_time / 5))


@contextmanager
def timelog(msg, min_log_time=1) -> float:
    """ "Measure and log time with loguru as context manager."""
    start = time.perf_counter()
    end = None
    thread_timer = ElapsedTimeThread(msg=msg, min_log_time=min_log_time)
    thread_timer.start()
    try:
        yield
        end = time.perf_counter() - start
    finally:
        thread_timer.stop()
        thread_timer.join()
        if end is None:
            logger.opt(colors=True).exception(f"{msg} failed")

    if end > min_log_time:
        logger.opt(colors=True).success(
            f"{msg} done" f"<green> 🕑 {round(end, 3)}sec</green>"
        )

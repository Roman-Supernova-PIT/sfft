import signal
import threading, ctypes

__author__ = "Lei Hu <leihu@andrew.cmu.edu>"
__version__ = "v1.2"

class Timeout():
    """ Timeout class using ALARM signal """
    class Timeout(Exception):
        pass
    def __init__(self, sec):
        self.sec = sec
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.raise_timeout)
        signal.alarm(self.sec)
    def __exit__(self, *args):
        signal.alarm(0)  # disable alarm
    def raise_timeout(self, *args):
        raise Timeout.Timeout()


class TimeoutAfter:
    """A time limit for a block of code, in the main thread or any other:

        with TimeoutAfter(timeout=300, exception=TimeoutError):
            ...

    When the block runs longer than `timeout` seconds, `exception` (an exception class) is raised in the thread that entered it. Unlike
    Timeout (an alarm signal, which only the main thread receives), it works in worker threads. The exception is delivered between two
    Python steps of that thread, so it waits for a long call into compiled code (numpy, an FFT) to return. Leaving the block in time
    cancels the limit, and the exception never arrives after the block has ended: if the time runs out while the block is ending, it is
    raised at the end of the `with` statement."""

    def __init__(self, timeout=10, exception=TimeoutError):
        self.timeout = timeout
        self.exception = exception
        self._lock = threading.Lock()
        self._running = False
        self._timer = None
        self._thread_id = None

    def __enter__(self):
        self._thread_id = threading.get_ident()
        self._running = True
        self._timer = threading.Timer(self.timeout, self._expire)
        self._timer.daemon = True
        self._timer.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        with self._lock:                     # after this, _expire() cannot raise into this thread any more
            self._running = False
        self._timer.cancel()
        return False

    def _expire(self):
        with self._lock:
            if not self._running:
                return
            self._running = False
            tid = ctypes.c_ulong(self._thread_id)
            hit = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(self.exception))
            if hit > 1:                      # never expected (one thread per id): undo rather than raise into several threads
                ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)

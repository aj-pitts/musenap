import sys
import time
import threading

class ProgressWheel:
    def __init__(self, message="Computing"):
        self.message = message
        self.spinning = False
        self._thread = None

    def _spin(self):
        frames = ['\\', '|', '/', '-']  # or use ['.', '..', '...', '..'] for dots
        i = 0
        while self.spinning:
            frame = frames[i % len(frames)]
            sys.stdout.write(f'\r{self.message} {frame}')
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1
        # Clear the line when done
        sys.stdout.write(f'\r\n')
        sys.stdout.flush()

    def start(self):
        self.spinning = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self):
        self.spinning = False
        self._thread.join()

    # Allows use as a context manager
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
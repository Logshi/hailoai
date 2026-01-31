"""
Threaded RTSP / camera frame capture.

Runs in a background thread, continuously grabs frames and pushes them
into a bounded queue. Old frames are discarded when the queue is full,
ensuring the consumer always gets the most recent frame.
"""

import threading
import time
import queue
import cv2
import config


class FrameCapture:
    def __init__(self, source=None):
        self.source = source if source is not None else config.RTSP_URL
        self._queue = queue.Queue(maxsize=config.CAPTURE_QUEUE_SIZE)
        self._stop_event = threading.Event()
        self._thread = None
        self._cap = None
        self.alive = False  # True once at least one frame has been read

    # -- public API --

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="capture")
        self._thread.start()

    def read(self, timeout=2.0):
        """Return the latest frame or None on timeout."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._release()

    # -- internals --

    def _open(self):
        """Open the video source with settings tuned for RTSP reliability."""
        self._release()
        # Use FFMPEG backend for RTSP; it handles reconnects better than GStreamer on Debian.
        cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        # Reduce RTSP buffering to minimize latency.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            return False
        self._cap = cap
        return True

    def _release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _run(self):
        """Main capture loop. Reconnects automatically on failure."""
        while not self._stop_event.is_set():
            if not self._open():
                print(f"[capture] Cannot open {self.source}. "
                      f"Retrying in {config.CAPTURE_RECONNECT_DELAY}s...")
                time.sleep(config.CAPTURE_RECONNECT_DELAY)
                continue

            print(f"[capture] Connected to {self.source}")
            consecutive_failures = 0

            while not self._stop_event.is_set():
                ok, frame = self._cap.read()
                if not ok:
                    consecutive_failures += 1
                    if consecutive_failures > 30:
                        print("[capture] Too many read failures, reconnecting...")
                        break
                    time.sleep(0.01)
                    continue

                consecutive_failures = 0
                self.alive = True

                # Drop old frame if queue is full (keep latency low).
                if self._queue.full():
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        pass
                self._queue.put(frame)

            self._release()
            if not self._stop_event.is_set():
                time.sleep(config.CAPTURE_RECONNECT_DELAY)

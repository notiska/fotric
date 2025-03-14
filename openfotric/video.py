#!/usr/bin/env python3

__all__ = (
    "next_file",
    "VideoWriter",
)

import os
import time
from queue import Queue
from threading import Thread

import cv2
import numpy as np


def next_file(parent: str, ext: str = ".mp4") -> str:
    """
    Returns the path of the next available video file.
    """

    prefix = 0

    for file in os.listdir(parent):
        if not file.startswith("out") or not file.endswith(ext):
            continue
        file, _ = os.path.splitext(file)
        if file != "out":
            try:
                number = int(file[3:]) + 1
            except ValueError:
                continue
        else:
            number = 1
        if number > prefix:
            prefix = number

    if not prefix:
        return os.path.join(parent, f"out{ext}")
    return os.path.join(parent, f"out{prefix}{ext}")


class VideoWriter:
    """
    A buffered video writer.

    Parameters
    ----------
    running: bool
        Whether this video writer is currently running.
    writing: bool
        Whether this video writer is currently writing.
    file: str
        The path of the file to write to.
    fourcc: int
        The fourcc code of the codec to use.
    fps: float
        The video framerate to write at.
    size: tuple[int, int]
        The size of the video frames.
    max_buffer: int
        The maximum number of frames to buffer.
    """

    __slots__ = ("file", "fourcc", "fps", "size", "max_buffer", "_thread", "_queue")

    @property
    def running(self) -> bool:
        return self._thread is not None

    @property
    def writing(self) -> bool:
        return not self._queue.empty()

    def __init__(self, file: str, fourcc: int, fps: float, size: tuple[int, int], max_buffer: int = 25) -> None:
        self.file = file
        self.fourcc = fourcc
        self.fps = fps
        self.size = size
        self.max_buffer = max_buffer

        self._thread: Thread | None = None
        self._queue = Queue()

    def _run(self) -> None:
        writer = cv2.VideoWriter(self.file, self.fourcc, self.fps, self.size)
        expect = 1 / self.fps
        last = time.time()

        # Waiting for first frame.
        while self._queue.empty():
            ...

        while self._thread is not None or not self._queue.empty():
            current, frame = self._queue.get()
            if frame is None:
                break
            while last < current:
                writer.write(frame)
                last += expect

        writer.release()

    def start(self) -> None:
        """
        Starts this video writer.
        """

        if self._thread is not None:
            return
        self._thread = Thread(target=self._run)
        self._thread.start()

    def stop(self):
        """
        Stops this video writer.
        """

        if self._thread is None:
            return
        self.write(None)
        # thread = self._thread
        self._thread = None
        # thread.join()

    def write(self, image: np.ndarray | None) -> None:
        """
        Writes a frame to the video.

        Parameters
        ----------
        image: np.ndarray | None
            The image to write, or `None` to stop the writer.
        """

        if self._thread and self._queue.qsize() < self.max_buffer:
            return self._queue.put((time.time(), image))

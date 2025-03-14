#!/usr/bin/env python3

import time
from threading import Thread

import cv2
import numpy as np

from .device import Device
from .image import Processor
from .video import next_file, VideoWriter

WIDTH = 768
HEIGHT = 576
FONT = (cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1)

running = True
device = None
frame = None


def run() -> None:
    """
    Background thread for processing the camera data.
    """

    global device
    global frame

    while running:
        try:
            if device is None:
                frame = None
                device = Device.find()
            elif not device.connected:
                frame = None
                device.connect()
            else:
                frame = device.frame()
        except Exception:
            frame = None
            device = None
            time.sleep(0.1)


def main() -> None:
    """
    Main thread.
    """

    global device
    global running

    cv2.namedWindow("FOTRIC")

    try:
        device = Device.find()
    except ValueError:
        ...
    processor = Processor(
        FONT, steps=[
            Processor.Gain(),
            Processor.Gamma(),
            Processor.BM3D(False),
            Processor.NLMeans(False),
            Processor.CLAHE(False),
            Processor.UpScale(WIDTH, HEIGHT, 3),
            Processor.UnsharpMask(True),
            Processor.ColourMap(None, True),
        ],
    )

    last_time = time.time()

    thread = Thread(target=run, daemon=True)
    thread.start()

    vw: VideoWriter | None = None
    debounce = False

    try:
        while running:
            if vw is not None and not vw.running and not vw.writing:
                vw = None
            capture = False

            delta = time.time() - last_time
            if delta < 0.033:
                time.sleep(0.033 - delta)
            last_time = time.time()

            if frame is not None:
                image = processor.process(frame)
            else:
                image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)

                if device is None:
                    text = "[NO DEVICE]"
                elif not device.connected:
                    text = "[CONNECTING]"
                else:
                    text = "[NO FRAME]"

                (width, height), _ = cv2.getTextSize(text, *FONT[:2], FONT[3])
                cv2.putText(image, text, (WIDTH // 2 - width // 2, HEIGHT // 2 - height // 2), *FONT)

            if frame is not None:
                if not capture and not debounce and not frame.flags & 1:
                    capture = True
                    debounce = True
                elif frame.flags & 1:
                    debounce = False

            if image.shape[0] != HEIGHT or image.shape[1] != WIDTH:
                image = cv2.resize(image, (WIDTH, HEIGHT))
            if vw is not None:
                (width, height), _ = cv2.getTextSize(f"REC {vw.file}", *FONT[:2], FONT[3])
                cv2.putText(image, f"REC {vw.file}", (WIDTH - width - 2, height + 2), *FONT)
                vw.write(image)
            cv2.imshow("FOTRIC", image)

            key = cv2.waitKeyEx(1)
            if key in (27, ord("q")):
                running = False
            elif key == ord(" "):
                if device is not None:
                    device.calibrate()
            elif key == ord("t"):
                if device is not None and frame is not None:
                    device.set_auto_cal(not frame.auto_cal)
            elif key == ord("y"):
                if device is not None and frame is not None:
                    device.set_nuc(not frame.nuc)
            elif key == ord("r"):
                if device is not None:
                    device.restart()
                device = None
            elif key == ord("d") or capture:
                if vw is None:
                    vw = VideoWriter(next_file(".", ".mp4"), cv2.VideoWriter.fourcc(*"mp4v"), 30, (WIDTH, HEIGHT))
                    vw.start()
                else:
                    vw.stop()
            else:
                processor.key(key)

    except KeyboardInterrupt:
        ...

    if device is not None:
        device.disconnect()


main()

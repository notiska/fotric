#!/usr/bin/env python3

__all__ = (
    "Frame",
)

import struct
from os import SEEK_SET
from typing import IO

import numpy as np


class Frame:
    """
    A single IR frame, also contains device status info.
    """

    __slots__ = (
        "state", "charge_state", "usb_state", "battery",
        "flags",
        "power_save", "auto_off", "auto_cal",
        "fid", "vsk", "gain", "int", "nuc",
        "lens_idx", "range_idx",
        "width", "height",
        "sensor_sn", "hw_version", "fw_version", "device_type", "sn", "time",
        "ad_values",
    )

    @classmethod
    def read(cls, stream: IO[bytes]) -> "Frame":
        """
        Reads the frame data from a binary stream.
        """

        self = cls()
        self.refresh(stream)
        return self

    @property
    def charging(self) -> bool:
        return bool(self.flags & 2)

    def __init__(self) -> None:
        self.state = 0
        self.charge_state = 0
        self.usb_state = 0
        self.battery = 0

        self.flags = 0

        self.power_save = False
        self.auto_off = False
        self.auto_cal = False

        self.lens_idx = 0
        self.range_idx = 0

        self.fid = 0
        self.vsk = 0
        self.gain = 0
        self.int = 0
        self.nuc = True

        self.sensor_sn = ""
        self.hw_version = ""
        self.fw_version = ""
        self.device_type = ""
        self.sn = ""
        self.time = ""

        self.ad_values: np.ndarray | None = None

    def __repr__(self) -> str:
        return (
            f"<Frame(state={self.state}, charge_state={self.charge_state}, usb_state={self.usb_state}, " +
            f"battery={self.battery}, flags={self.flags}, power_save={self.power_save}, auto_off={self.auto_off}, " +
            f"auto_cal={self.auto_cal}, sensor_sn={self.sensor_sn!r}, width={self.width}, height={self.height}, " +
            f"hw_version={self.hw_version!r}, fw_version={self.fw_version!r}, device_type={self.device_type!r}, " +
            f"sn={self.sn!r}, time={self.time!r})>"
        )

    def refresh(self, stream: IO[bytes]) -> None:
        """
        Reads the frame data from a binary stream.
        """

        assert stream.seekable(), "stream is not seekable"

        stream.seek(8, SEEK_SET)
        self.state, self.charge_state, self.usb_state, self.battery = stream.read(4)

        stream.seek(15, SEEK_SET)
        self.flags, = stream.read(1)

        stream.seek(17, SEEK_SET)
        self.power_save, = struct.unpack("?", stream.read(1))
        stream.seek(20, SEEK_SET)
        self.auto_off, self.auto_cal = struct.unpack("??", stream.read(2))

        stream.seek(78, SEEK_SET)
        self.fid, self.vsk, self.gain, self.int, self.nuc = struct.unpack(">hhbh?", stream.read(8))

        stream.seek(832, SEEK_SET)
        self.sensor_sn   = stream.read(16).decode("utf-8", errors="ignore").rstrip("\x00")
        stream.seek(858, SEEK_SET)
        self.width, self.height = struct.unpack(">hh", stream.read(4))
        # self.proto_info  = stream.read(16)
        stream.seek(864, SEEK_SET)
        self.hw_version  = stream.read(16).decode("utf-8", errors="ignore").rstrip("\x00")
        stream.seek(896, SEEK_SET)
        self.fw_version  = stream.read(16).decode("utf-8", errors="ignore").rstrip("\x00")
        stream.seek(944, SEEK_SET)
        self.device_type = stream.read(16).decode("utf-8", errors="ignore").rstrip("\x00")
        stream.seek(976, SEEK_SET)
        self.sn          = stream.read(16).decode("utf-8", errors="ignore").rstrip("\x00")
        stream.seek(1008, SEEK_SET)
        self.time        = stream.read(16).decode("utf-8", errors="ignore").rstrip("\x00")

        stream.seek(2048, SEEK_SET)
        ad_values = np.frombuffer(stream.read(self.height * 2048), dtype=np.dtype("<H"))
        ad_values = ad_values.reshape((self.height, 1024))
        self.ad_values = ad_values[:, :self.width]

#!/usr/bin/env python3

__all__ = (
    "Params",
)

import struct
from os import SEEK_SET
from typing import IO, Iterable, Self

import numpy as np


class Params:
    """
    Core device parameters
    """

    __slots__ = (
        "width", "height", "lens_idx", "range_idx",
        "ad_table", "temp_table",
        "id", "sn0", "sn1", "model", "version",
        "hw", "hw_version", "sensor", "sensor_sn", "fpga_version",
        "tag0", "tag1", "tag2",
        "sealed", "seal_date", "create_date", "modify_date", "updates",
        "param_version", "lenses",
        "crop_width", "crop_height", "max_fps", "vid", "max_regions", "max_points", "max_lines",
    )

    @classmethod
    def read(cls, stream: IO[bytes]) -> Self:
        """
        Reads core device parameters from a binary stream.

        Raises
        ------
        ValueError
            If the parameters could not be read.
        """

        self = cls()
        if not self.refresh(stream):
            raise ValueError("failed to read parameters")
        return self

    def __init__(self) -> None:
        self.width = 384
        self.height = 288
        self.lens_idx = 0
        self.range_idx = 0

        self.ad_table: np.ndarray | None = None
        self.temp_table: np.ndarray | None = None

        self.id = ""
        self.sn0 = ""
        self.sn1 = ""
        self.model = ""
        self.version = ""

        self.hw = ""
        self.hw_version = ""
        self.sensor = ""
        self.sensor_sn = ""
        self.fpga_version = ""

        self.tag0 = ""
        self.tag1 = ""
        self.tag2 = ""

        self.sealed = False
        self.seal_date = ""
        self.create_date = ""
        self.modify_date = ""
        self.updates = 0

        self.param_version = ""
        self.lenses: list[Params.Lens] = []

        self.crop_width = 0
        self.crop_height = 0
        self.max_fps = 0
        self.vid = 0
        self.max_regions = 0
        self.max_points = 0
        self.max_lines = 0

    def __repr__(self) -> str:
        return (
            f"<Params(id={self.id!r}, sn0={self.sn0!r}, sn1={self.sn1!r}, model={self.model!r}, " +
            f"version={self.version!r}, hw={self.hw!r}, hw_version={self.hw_version!r}, sensor={self.sensor!r}, " +
            f"sensor_sn={self.sensor_sn!r}, fpga_version={self.fpga_version!r}, tag0={self.tag0!r}, " + 
            f"tag1={self.tag1!r}, tag2={self.tag2!r}, sealed={self.sealed}, seal_date={self.seal_date!r}, " +
            f"create_date={self.create_date!r}, modify_date={self.modify_date!r}, updates={self.updates}, " +
            f"param_version={self.param_version!r}, lenses={self.lenses!r}, crop_width={self.crop_width}, " +
            f"crop_height={self.crop_height}, max_fps={self.max_fps}, vid=0x{self.vid:04x}, " +
            f"max_regions={self.max_regions}, max_points={self.max_points}, max_lines={self.max_lines})>"
        )

    def refresh(self, stream: IO[bytes]) -> bool:
        """
        Refreshes these core device parameters from a binary stream.
        """

        assert stream.seekable(), "stream is not seekable"

        # stream.seek(8, SEEK_SET)
        # state, = stream.read(1)
        # print(state)
        # if state:
        #     return False

        stream.seek(25, SEEK_SET)
        self.lens_idx, self.range_idx = stream.read(2)

        stream.seek(858, SEEK_SET)
        self.width, self.height = struct.unpack(">hh", stream.read(4))

        if self.width < 1 or self.height < 1:
            return False

        stream.seek(1410, SEEK_SET)
        table_size, = struct.unpack(">h", stream.read(2))
        if 0 < table_size < 64:
            stream.seek(1284, SEEK_SET)
            self.ad_table = np.frombuffer(stream.read(table_size * 2), dtype=np.dtype(">H"))
            stream.seek(1412, SEEK_SET)
            self.temp_table = np.frombuffer(stream.read(table_size * 2), dtype=np.dtype(">h")) / 10

        stream.seek(2048, SEEK_SET)
        self.id      = stream.read(80).decode("utf-8", errors="ignore").rstrip("\x00")
        self.sn0     = stream.read(80).decode("utf-8", errors="ignore").rstrip("\x00")
        self.sn1     = stream.read(80).decode("utf-8", errors="ignore").rstrip("\x00")
        self.model   = stream.read(80).decode("utf-8", errors="ignore").rstrip("\x00")
        self.version = stream.read(80).decode("utf-8", errors="ignore").rstrip("\x00")

        self.hw           = stream.read(80).decode("utf-8", errors="ignore").rstrip("\x00")
        self.hw_version   = stream.read(80).decode("utf-8", errors="ignore").rstrip("\x00")
        self.sensor       = stream.read(80).decode("utf-8", errors="ignore").rstrip("\x00")
        self.sensor_sn    = stream.read(80).decode("utf-8", errors="ignore").rstrip("\x00")
        self.fpga_version = stream.read(80).decode("utf-8", errors="ignore").rstrip("\x00")

        self.tag0 = stream.read(80).decode("utf-8", errors="ignore").rstrip("\x00")
        self.tag1 = stream.read(80).decode("utf-8", errors="ignore").rstrip("\x00")
        self.tag2 = stream.read(80).decode("utf-8", errors="ignore").rstrip("\x00")

        sealed, = struct.unpack("<I", stream.read(4))
        self.sealed = bool(sealed)
        self.seal_date   = stream.read(16).decode("utf-8", errors="ignore").rstrip("\x00")
        self.create_date = stream.read(16).decode("utf-8", errors="ignore").rstrip("\x00")
        self.modify_date = stream.read(16).decode("utf-8", errors="ignore").rstrip("\x00")
        self.updates, = struct.unpack("<I", stream.read(4))

        stream.seek(4096, SEEK_SET)
        self.param_version = stream.read(80).decode("utf-8", errors="ignore").rstrip("\x00")

        lens_count, = struct.unpack("<I", stream.read(4))
        self.lenses.clear()
        if lens_count <= 10:
            base = stream.tell()
            for index in range(lens_count):
                stream.seek(base + index * 124, SEEK_SET)
                name = stream.read(60).decode("utf-8", errors="ignore").rstrip("\x00")
                tag  = stream.read(20).decode("utf-8", errors="ignore").rstrip("\x00")
                ranges_count, = struct.unpack("<I", stream.read(4))
                ranges = []
                for _ in range(ranges_count):
                    min_, max_ = struct.unpack("<ii", stream.read(8))
                    ranges.append(Params.Range(min_, max_))
                self.lenses.append(Params.Lens(name, tag, ranges))

        stream.seek(6144, SEEK_SET)
        self.crop_width, self.crop_height, self.max_fps = struct.unpack("<III", stream.read(12))
        self.vid, = struct.unpack("<I", stream.read(4))
        self.max_regions, self.max_points, self.max_lines = struct.unpack("<III", stream.read(12))

        return True

    class Lens:
        """
        Information about a lens.
        """

        __slots__ = ("name", "tag", "ranges")

        def __init__(self, name: str, tag: str, ranges: Iterable["Params.Range"] | None = None) -> None:
            self.name = name
            self.tag = tag
            self.ranges: list[Params.Range] = []

            if ranges is not None:
                self.ranges.extend(ranges)

        def __repr__(self) -> str:
            return f"<Params.Lens(name={self.name!r}, tag={self.tag!r}, ranges={self.ranges!r})>"

    class Range:
        """
        Information about a supported temperature range.
        """

        __slots__ = ("min", "max")

        def __init__(self, min_: int, max_: int) -> None:
            self.min = min_
            self.max = max_

        def __repr__(self) -> str:
            return f"<Params.Range(min={self.min}, max={self.max})>"

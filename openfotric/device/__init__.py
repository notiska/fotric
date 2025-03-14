#!/usr/bin/env python3

__all__ = (
    "frame", "params",
    "Frame", "Params",
    "Device",
)

from functools import cache
from io import BytesIO
from threading import RLock

from usb import core
from usb.core import Device as USBDevice, Endpoint, USBError

from . import frame, params
from .frame import *
from .params import *


class Device:
    """
    The Fotric device handle.
    """

    __slots__ = (
        "_handle", "_cmd_bulk_out", "_cmd_bulk_in", "_data_bulk_out", "_data_bulk_in",
        "connected", "_connecting", "_lock",
        "params",
    )

    VID = 0x0547
    PID = 0x0503

    @classmethod
    def find(cls) -> "Device":
        """
        Attempts to find the Fotric device.

        Raises
        ------
        ValueError
            If the camera could not be found.
        """

        handle = core.find(idVendor=cls.VID, idProduct=cls.PID)
        if handle is None:
            raise ValueError("failed to find USB device")
        return cls(handle)

    def __init__(self, handle: USBDevice) -> None:
        self._handle = handle

        self._cmd_bulk_out:  Endpoint | None = None
        self._cmd_bulk_in:   Endpoint | None = None
        self._data_bulk_out: Endpoint | None = None
        self._data_bulk_in:  Endpoint | None = None

        # self._handle.set_configuration()
        config = self._handle.get_active_configuration()
        if config is None:
            # TODO: Can we select an active configuration?
            raise ValueError("USB device has no active configuration")

        interface = config[0, 0]
        try:
            (
                self._cmd_bulk_out,
                self._cmd_bulk_in,
                self._data_bulk_out,
                self._data_bulk_in,
            ) = interface.endpoints()
        except ValueError:
            raise ValueError("failed to find all USB endpoints")

        self.connected = False
        self._connecting = False
        self._lock = RLock()

        self.params = Params()

    def connect(self) -> bool:
        """
        Attempts to connect to the device.
        """

        if self.connected:
            return True

        with self._lock:
            if not self._connecting:
                self._cmd_bulk_out.write(Device.Command.pack(True, 0, Device.Command.SYSTEM_MODE, b"\x6e\x01\x38\x04"))
                self._cmd_bulk_out.write(Device.Command.pack(False, 0, Device.Command.SYSTEM_MODE))

                self._cmd_bulk_out.write(Device.Command.pack(True, 0, Device.Command.READ_IR_FRAME))
                self._cmd_bulk_out.write(Device.Command.pack(False, 0, Device.Command.READ_IR_FRAME))

            self._cmd_bulk_out.write(Device.Command.pack(True, 4, Device.Command.READ_IR_FRAME))
            stream = BytesIO(self._data_bulk_in.read(12288))
            self._cmd_bulk_out.write(Device.Command.pack(False, 4, Device.Command.READ_IR_FRAME))

            if not self.params.refresh(stream):
                self._connecting = True
                return False
            self.connected = True
            self._connecting = False

        return True

    def frame(self) -> Frame:
        """
        Reads the next frame from the device.
        """

        with self._lock:
            self._cmd_bulk_out.write(Device.Command.pack(True, 0, Device.Command.READ_IR_FRAME))
            stream = BytesIO(self._data_bulk_in.read(2048 + self.params.height * 2048))
            self._cmd_bulk_out.write(Device.Command.pack(False, 0, Device.Command.READ_IR_FRAME))

        return Frame.read(stream)

    def calibrate(self) -> None:
        """
        Performs a manual calibration.
        """

        with self._lock:
            self._cmd_bulk_out.write(Device.Command.pack(True, 0, Device.Command.DO_CALIBRATE))
            self._cmd_bulk_out.write(Device.Command.pack(False, 0, Device.Command.DO_CALIBRATE))

    def set_auto_cal(self, value: bool) -> None:
        """
        Sets the camera auto-calibrate parameter.
        """

        with self._lock:
            self._cmd_bulk_out.write(Device.Command.pack(True, 0, Device.Command.AUTO_CALIBRATE, bytes([value])))
            self._cmd_bulk_out.write(Device.Command.pack(False, 0, Device.Command.AUTO_CALIBRATE, bytes([value])))

    def set_nuc(self, value: bool) -> None:
        """
        Sets the camera NUC parameter.
        """

        with self._lock:
            self._cmd_bulk_out.write(Device.Command.pack(True, 0, Device.Command.NUC_ENABLED, bytes([value])))
            self._cmd_bulk_out.write(Device.Command.pack(False, 0, Device.Command.NUC_ENABLED, bytes([value])))

    def restart(self) -> None:
        """
        Restarts the camera.
        """

        with self._lock:
            self._cmd_bulk_out.write(Device.Command.pack(True, 0, Device.Command.POWER_OFF, b"\x7f\x01\x08\x46"))
            self._cmd_bulk_out.write(Device.Command.pack(False, 0, Device.Command.POWER_OFF, b"\x7f\x01\x08\x46"))

        self.disconnect()

    def disconnect(self) -> None:
        """
        Disconnects from the device.
        """

        if not self.connected:
            return
        self.connected = False
        try:
            self._handle.reset()
        except USBError:
            ...

    # ------------------------------ Classes ------------------------------ #

    class Command:
        """
        USB device commands.
        """

        READ_IR_FRAME     = 0
        READ_BAD_POINT    = 0
        DO_CALIBRATE      = 1
        AUTO_CALIBRATE    = 2
        SELECT_LENS       = 6
        SELECT_TEMP_RANGE = 5

        SYSTEM_MODE       = 96

        NUC               = 101
        NUC_ENABLED       = 106

        SAVE              = 111

        SET_BASE_TEMP     = 115
        SET_LUT           = 114
        SET_SYSTEM_PARAM  = 116

        POWER_OFF         = 127

        @classmethod
        @cache
        def pack(cls, start: bool, id_: int, command: int, data: bytes | None = None) -> bytes:
            """
            Packs the provided command into bytes to be sent over the wire.

            Parameters
            ----------
            start: bool
                `True` if this is the start command start indicator, otherwise `False`.
            id_: int
                The command ID.
            command: int
                The command key, see this class' attributes.
            data: bytes | None
                Any parameter data to send, or `None` if no data or the end command.
            """

            header = bytes([
                85 if start else 94,    # flag0
                171 if start else 170,  # flag1
                id_,                    # command
                0,                      # bitsel
                255,                    # reserved
                command,                # key_command
            ])
            if data is None:
                return header + b"\x00" * 50
            return header + data + b"\x00" * (50 - len(data))

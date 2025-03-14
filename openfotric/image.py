#!/usr/bin/env python3

__all__ = (
    "Processor",
)

from enum import Enum
from typing import Iterable

import cv2
import numpy as np
import pybm3d

from .device import Frame


class Processor:
    """
    The image processor.

    Attributes
    ----------
    steps: list[Processor.Step]
        The processing steps.
    hud: bool
        Whether to display the HUD.
    scaling: Processor.Scaling
        The scaling mode to use.
    gain: float
        Scaling gain factor.
    alpha: float
        Temporal smoothing factor for scaling.
    """

    __slots__ = (
        "_font",
        "steps", "hud", "scaling", "alpha", "gain",
        "lock", "_min", "_max",
    )

    def __init__(
            self, font: tuple[int, float, tuple[int, int, int], int],
            steps: Iterable["Processor.Step"] | None = None,
    ) -> None:
        self._font = font

        self.steps: list[Processor.Step] = []
        self.hud = True
        self.scaling = Processor.Scaling.ADAPTIVE
        self.alpha = 0.1
        self.gain = 1.0

        if steps is not None:
            self.steps.extend(steps)

        self.lock = False
        self._min: float | None = None
        self._max: float | None = None

    def process(self, frame: Frame) -> np.ndarray:
        """
        Processes the raw frame from the camera.

        Parameters
        ----------
        frame: Frame
            The frame data to process.

        Returns
        -------
        np.ndarray
            An 8-bit colour image.
        """

        for step in self.steps:
            if step.enabled:
                step.raw(self, frame)
        data = frame.ad_values

        if self.lock and self._min is not None and self._max is not None:
            lower = self._min
            upper = self._max
        elif self.scaling is Processor.Scaling.ADAPTIVE:
            lower = np.percentile(data, 2)
            upper = np.percentile(data, 98)
        else:
            lower = np.min(data)
            upper = np.max(data)

        if not self.lock and self._min is not None and self._max is not None:
            lower = self.alpha * lower + (1 - self.alpha) * self._min
            upper = self.alpha * upper + (1 - self.alpha) * self._max

        self._min = lower
        self._max = upper

        dyn_range = upper - lower

        if self.scaling is Processor.Scaling.LINEAR or self.scaling is Processor.Scaling.ADAPTIVE:
            image = np.clip((data - lower) * (255 / dyn_range) * self.gain, 0, 255).astype(np.uint8)
        elif self.scaling is Processor.Scaling.LOG:
            data_norm = (np.clip(data, lower, upper) - lower) / dyn_range
            log_base = 1 + max(self.gain, 1e-6)
            log_data = np.log1p(max(self.gain, 1e-6) * data_norm) / np.log(log_base)
            image = (log_data * 255).clip(0, 255).astype(np.uint8)
        elif self.scaling is Processor.Scaling.HISTOGRAM:
            # https://stackoverflow.com/questions/70442513/why-does-histogram-equalization-on-a-16-bit-image-show-a-strange-result
            hist, bins = np.histogram(data, bins=256, range=(lower, upper))
            cdf = hist.cumsum()
            cdf = cdf / cdf[-1]
            cdf_mod = cdf ** self.gain
            image = np.interp(data, bins[:-1], cdf_mod * 255).astype(np.uint8)
        else:
            raise NotImplementedError(f"scaling mode {self.scaling!r} is not implemented")

        intermediaries = {}
        for step in self.steps:
            if not step.enabled:
                continue
            image = step.process(image)
            intermediaries[step.name] = image

        # image = cv2.resize(image, (self._width, self._height))
        image_height, image_width, *_ = image.shape
        if self.hud:
            compact = ["_A"[frame.auto_cal], "_N"[frame.nuc], "_L"[self.lock], "_H"[self.hud]]
            compact.extend(("_", step.compact)[step.enabled] for step in self.steps if step.compact is not None)
            bottom_left = [
                "".join(compact),
                f"{self.scaling.name} (x{self.gain:.1f} {lower:.1f}-{upper:.1f})",
            ]
            bottom_right = [
                f"DISCHARGING ({frame.battery}%)" if not frame.charging else "CHARGING",
                f"{frame.hw_version} {frame.fw_version}",
                f"{frame.device_type} ({frame.sn})",
            ]

            for index, step in enumerate(filter(lambda step: step.compact is None and step.enabled, self.steps)):
                step_str = str(step)
                (_, text_height), _ = cv2.getTextSize(step_str, *self._font[:2], self._font[3])
                pos = (1, text_height * (index + 1) + 2 + index)
                cv2.putText(image, step_str, pos, *self._font)

            for index, text in enumerate(reversed(bottom_left)):
                (text_width, text_height), _ = cv2.getTextSize(text, *self._font[:2], self._font[3])
                pos = (1, image_height - text_height * index - (2 + index))
                cv2.putText(image, text, pos, *self._font)
            for index, text in enumerate(reversed(bottom_right)):
                (text_width, text_height), _ = cv2.getTextSize(text, *self._font[:2], self._font[3])
                pos = (image_width - text_width - 1, image_height - text_height * index - (2 + index))
                cv2.putText(image, text, pos, *self._font)

        return image

    def key(self, code: int) -> None:
        """
        Processes a key press.
        """

        if code == ord("h"):
            self.hud = not self.hud
        elif code == ord("l"):
            self.lock = not self.lock
        elif code == ord("m"):
            self.scaling = Processor.Scaling((self.scaling.value + 1) % len(Processor.Scaling))

        for step in self.steps:
            step.key(code)

    # ------------------------------ Classes ------------------------------ #

    class Scaling(Enum):
        """
        The method used to scale the 16-bit data to 8-bit.

        Attributes
        ----------
        LINEAR: int
            Linear scaling.
        ADAPTIVE: int
            Adaptive linear scaling.
        LOG: int
            Logarithmic scaling.
        HISTOGRAM: int
            Global histogram equalisation.
        """

        LINEAR    = 0
        ADAPTIVE  = 1
        LOG       = 2
        HISTOGRAM = 3
        # CLAHE       = "CLAHE"

    class Step:
        """
        An image processing step.

        Attributes
        ----------
        name: str
            The name of the step.
        compact: str | None
            A single letter to display on the HUD.
            If `None`, the full str() call of the step is used.
        enabled: bool
            Whether this processing step is enabled.
        """

        __slots__ = ("enabled",)

        name: str
        compact: str | None = None

        def __init__(self, enabled: bool) -> None:
            self.enabled = enabled

        def __repr__(self) -> str:
            raise NotImplementedError(f"repr() is not implemented for {type(self)!r}")

        def __str__(self) -> str:
            # raise NotImplementedError(f"str() is not implemented for {type(self)!r}")
            return self.name

        def raw(self, processor: "Processor", frame: Frame) -> None:
            """
            Accepts the raw frame data, to do any pre-processing.
            """

            raise NotImplementedError(f"raw() is not implemented for {type(self)!r}")

        def process(self, image: np.ndarray) -> np.ndarray:
            """
            Applies this step to the image.
            """

            raise NotImplementedError(f"process() is not implemented for {type(self)!r}")

        def key(self, code: int) -> None:
            """
            Processes a key press.
            """

            raise NotImplementedError(f"key() is not implemented for {type(self)!r}")

    class Gain(Step):
        """
        Applies gain to the image.
        """

        __slots__ = ("gain", "auto")

        name = "gain"
        compact = None

        def __init__(self, gain: float = 1.0, auto: bool = False) -> None:
            super().__init__(True)
            self.gain = gain
            self.auto = auto

        def __repr__(self) -> str:
            return f"<Gain(gain={self.gain:.1f}, auto={self.auto})>"

        def __str__(self) -> str:
            if self.auto:
                return f"Gain: {self.gain:.2f} (AUTO)"
            return f"Gain: {self.gain:.2f}"

        def raw(self, processor: "Processor", frame: Frame) -> None:
            if self.auto:
                # Determines the gain based on the std deviation of the raw frame. (ChatGPT moment.)
                std_dev = np.std(frame.ad_values)
                gain = 1.0 + (50.0 / (std_dev + 1e-5))
                self.gain = np.clip(gain, 0.5, 2.5)
            processor.gain = self.gain

        def process(self, image: np.ndarray) -> np.ndarray:
            # if self.gain == 1.0:
            #     return image
            # return np.clip(image * self.gain, 0, 255).astype(np.uint8)
            return image

        def key(self, code: int) -> None:
            if code == ord("a"):
                self.auto = not self.auto
            elif code in (ord("w"), 65362):  # Up key.
                self.gain = min(2.5, round(self.gain, 1) + 0.1)
                self.auto = False
            elif code in (ord("s"), 65364):  # Down key.
                self.gain = max(0.5, round(self.gain, 1) - 0.1)
                self.auto = False

    class Gamma(Step):
        """
        Applies gamma correction to the image.
        """

        __slots__ = ("gamma", "auto", "_lut", "_last_gamma")

        name = "gamma"
        compact = None

        def __init__(self, gamma: float = 1.0, auto: bool = False) -> None:
            super().__init__(True)
            self.gamma = gamma
            self.auto = auto

            self._lut: np.ndarray | None = None
            self._last_gamma: float | None = None

        def __repr__(self) -> str:
            return f"<Gamma(gamma={self.gamma:.2f}, auto={self.auto}>"

        def __str__(self) -> str:
            if self.auto:
                return f"Gamma: {self.gamma:.2f} (AUTO)"
            return f"Gamma: {self.gamma:.2f}"

        def raw(self, processor: "Processor", frame: Frame) -> None:
            if self.auto:
                # Determines the gamma based on mean intensity of the raw frame. (ChatGPT moment, again.)
                mean = np.mean(frame.ad_values)
                mean_norm = (mean - frame.ad_values.min()) / (frame.ad_values.max() - frame.ad_values.min() + 1e-5)
                gamma = 2.0 - mean_norm
                self.gamma = np.clip(gamma, 0.5, 3.5)

        def process(self, image: np.ndarray) -> np.ndarray:
            if self.gamma == 1.0:
                return image
            elif self._lut is None or self._last_gamma != self.gamma:
                gamma_inv = 1 / self.gamma
                self._lut = np.array([((value / 255.0) ** gamma_inv) * 255 for value in range(256)]).astype(np.uint8)
            return cv2.LUT(image, self._lut)

        def key(self, code: int) -> None:
            if code == ord("a"):
                self.auto = not self.auto
            elif code == ord("z"):
                self.gamma = max(0.5, round(self.gamma, 1) - 0.1)
                self.auto = False
            elif code == ord("x"):
                self.gamma = min(3.5, round(self.gamma, 1) + 0.1)
                self.auto = False

    class BM3D(Step):
        """
        Applies BM3D denoising to the image.
        """

        __slots__ = ()

        name = "bm3d"
        compact = "B"

        def raw(self, processor: "Processor", frame: Frame) -> None:
            ...

        def process(self, image: np.ndarray) -> np.ndarray:
            return pybm3d.bm3d.bm3d(image, 15)

        def key(self, code: int) -> None:
            if code == ord("b"):
                self.enabled = not self.enabled

    class NLMeans(Step):
        """
        Applies NL-means denoising to the image.
        """

        __slots__ = ()

        name = "nl_means"
        compact = "M"

        def raw(self, processor: "Processor", frame: Frame) -> None:
            ...

        def process(self, image: np.ndarray) -> np.ndarray:
            return cv2.fastNlMeansDenoising(image)

        def key(self, code: int) -> None:
            if code == ord("v"):  # FIXME: Terrible keybind.
                self.enabled = not self.enabled

    class CLAHE(Step):
        """
        Applies advanced CLAHE to the image.

        ChatGPT moment.
        """

        __slots__ = (
            "_clahe_list",
            "_clahe_weights",
            "_alpha",
            "_beta",
            "_prev_image",
            "_prev_processed",
            "_motion_threshold",
        )

        name = "CLAHE"
        compact = "C"

        def __init__(
                self, enabled: bool, clip_limit: float = 2.0,
                tile_grid_sizes: tuple[tuple[int, int], ...] = ((8, 8), (16, 16), (32, 32)),
                clahe_weights: tuple[float, ...] = (0.5, 0.3, 0.2),
                alpha: float = 0.7, beta: float = 0.3, motion_threshold: float = 15.0,
        ) -> None:
            super().__init__(enabled)
            self._clahe_list = [(size, cv2.createCLAHE(clip_limit, size)) for size in tile_grid_sizes]
            self._clahe_weights = clahe_weights  # Weights for combining CLAHE images
            self._alpha = alpha  # Blending factor with original image
            self._beta = beta  # Temporal smoothing factor
            self._motion_threshold = motion_threshold  # Threshold for motion detection
            self._prev_image: np.ndarray | None = None  # Previous original image
            self._prev_processed: np.ndarray | None = None  # Previous processed image

        def __repr__(self) -> str:
            return f"<CLAHE>"

        def raw(self, processor: "Processor", frame: Frame) -> None:
            pass  # Implement if needed

        def process(self, image: np.ndarray) -> np.ndarray:
            # Multi-Scale CLAHE
            clahe_images = []
            for (size, clahe), weight in zip(self._clahe_list, self._clahe_weights):
                # Apply CLAHE with current tile size
                clahe_image = clahe.apply(image)
                # Multiply by weight and store as float for accurate summation
                clahe_images.append(clahe_image.astype(np.float32) * weight)
            # Combine CLAHE images from different scales
            combined_clahe = sum(clahe_images)
            # Normalize to 0-255 and convert to uint8
            combined_clahe = np.clip(combined_clahe, 0, 255).astype(np.uint8)

            # Blending with Original Image
            blended_image = cv2.addWeighted(combined_clahe, self._alpha, image, 1 - self._alpha, 0)

            # Temporal Smoothing
            if self._prev_processed is not None and self._prev_image is not None:
                # Calculate Motion Mask
                diff = cv2.absdiff(image, self._prev_image)
                _, motion_mask = cv2.threshold(diff, self._motion_threshold, 255, cv2.THRESH_BINARY)
                # Optional: Apply morphological operations to clean up the mask
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
                motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
                # Normalize motion mask to range [0, 1]
                motion_mask = motion_mask.astype(np.float32) / 255.0
                # Compute static mask (areas with little or no motion)
                static_mask = 1.0 - motion_mask
                # Adjust beta based on motion
                beta_map = self._beta * static_mask
                beta_map = np.clip(beta_map, 0, self._beta)
                # Perform pixel-wise temporal smoothing
                smoothed_image = (blended_image.astype(np.float32) * (1 - beta_map) +
                                  self._prev_processed.astype(np.float32) * beta_map).astype(np.uint8)
            else:
                # If no previous frame, use the blended image as is
                smoothed_image = blended_image.copy()

            # Update previous images for the next frame
            self._prev_image = image.copy()
            self._prev_processed = smoothed_image.copy()

            return smoothed_image

        def key(self, code: int) -> None:
            if code == ord("c"):
                self.enabled = not self.enabled

    class UpScale(Step):
        """
        Up-scales the image.
        """

        __slots__ = (
            "width", "height", "interp",
            "_start_width", "_start_height",
        )

        name = "up_scale"
        compact = None

        _INTERPS = [
            "INTER_NEAREST",
            "INTER_LINEAR",
            "INTER_CUBIC",
            "INTER_LANCZOS4",
        ]

        def __init__(self, width: int, height: int, interp: int = 0) -> None:
            super().__init__(True)
            self.width = width
            self.height = height
            self.interp = interp

            self._start_width: int | None = None
            self._start_height: int | None = None

        def __repr__(self) -> str:
            return f"<UpScale(width={self.width}, height={self.height}, interp={self.interp})>"

        def __str__(self) -> str:
            interp_str = self._INTERPS[self.interp][len("INTER_"):]
            if self._start_width is None or self._start_height is None:
                return f"{self.width}x{self.height} ({interp_str})"
            return f"{self._start_width}x{self._start_height} ({interp_str})"

        def raw(self, processor: "Processor", frame: Frame) -> None:
            self._start_width = frame.width
            self._start_height = frame.height

        def process(self, image: np.ndarray) -> np.ndarray:
            return cv2.resize(image, (self.width, self.height), interpolation=getattr(cv2, self._INTERPS[self.interp]))

        def key(self, code: int) -> None:
            if code == ord("n"):
                self.interp = (self.interp + 1) % len(self._INTERPS)

    class UnsharpMask(Step):
        """
        Applies an unsharp mask to the image.
        """

        __slots__ = ()

        name = "unsharp_mask"
        compact = "U"

        def __repr__(self) -> str:
            return "<UnsharpMask>"

        def raw(self, processor: "Processor", frame: Frame) -> None:
            ...

        def process(self, image: np.ndarray) -> np.ndarray:
            return cv2.addWeighted(image, 1.5, cv2.GaussianBlur(image, (9, 9), 10.0), -0.5, 0)

        def key(self, code: int) -> None:
            if code == ord("u"):
                self.enabled = not self.enabled

    class ColourMap(Step):
        """
        Applies a colour map to the image.
        """

        __slots__ = ("map", "invert")

        name = "colour_map"
        compact = None

        _MAPS = [
            "COLORMAP_AUTUMN",
            "COLORMAP_BONE",
            "COLORMAP_JET",
            "COLORMAP_WINTER",
            "COLORMAP_RAINBOW",
            "COLORMAP_OCEAN",
            "COLORMAP_SUMMER",
            "COLORMAP_SPRING",
            "COLORMAP_COOL",
            "COLORMAP_HSV",
            "COLORMAP_PINK",
            "COLORMAP_HOT",
            "COLORMAP_PARULA",
            "COLORMAP_MAGMA",
            "COLORMAP_INFERNO",
            "COLORMAP_PLASMA",
            "COLORMAP_VIRIDIS",
            "COLORMAP_CIVIDIS",
            "COLORMAP_TWILIGHT",
            "COLORMAP_TWILIGHT_SHIFTED",
            "COLORMAP_TURBO",
            "COLORMAP_DEEPGREEN",
        ]

        def __init__(self, map_: int | None, invert: bool) -> None:
            super().__init__(True)
            self.map: int | None = map_
            self.invert = invert

        def __repr__(self) -> str:
            return f"<ColourMap(map={self.map}, invert={self.invert})>"

        def __str__(self) -> str:
            if self.map is None:
                return "WHITE" if not self.invert else "BLACK"
            map_str = self._MAPS[self.map][len("COLORMAP_"):]
            return f"{map_str} (INV)" if self.invert else map_str

        def raw(self, processor: "Processor", frame: Frame) -> None:
            ...

        def process(self, image: np.ndarray) -> np.ndarray:
            if self.map is not None and (self.map < 0 or self.map >= len(self._MAPS)):
                self.map = None
            if self.invert:
                image = cv2.bitwise_not(image)
            if self.map is None:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            return cv2.applyColorMap(image, getattr(cv2, self._MAPS[self.map]))

        def key(self, code: int) -> None:
            if code in (ord("]"), 65363):  # Right key.
                if self.map is None:
                    self.map = 0
                elif self.map < len(self._MAPS) - 1:
                    self.map += 1
                else:
                    self.map = None
            elif code in (ord("["), 65361):  # Left key.
                if self.map is None:
                    self.map = len(self._MAPS) - 1
                elif self.map > 0:
                    self.map -= 1
                else:
                    self.map = None
            elif code == ord("i"):
                self.invert = not self.invert

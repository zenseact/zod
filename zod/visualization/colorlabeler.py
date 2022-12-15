# (c) Copyright 2021-2022 Zenseact AB
"""Color labeler using some colormaps from OpenCV."""

from typing import Callable, Union

import cv2
import matplotlib as mpl

# pylint: disable=unused-import
import matplotlib.cm
import numpy as np


def create_matplotlib_colormap(src: np.ndarray, colormap_type: str) -> np.ndarray:
    """Create colormap image from matplotlib.

    Args:
        src : source image with specified shape
        colormap_type : colormap type

    Returns:
        array with colored values regarding specified colormap

    """
    distance_list = src[:, 0, 0]

    # use the coolwarm colormap that is built-in, and goes from blue to red
    cmap = getattr(mpl.cm, colormap_type)

    # convert your distances to color coordinates
    color_list = cmap(distance_list)
    return (np.array(color_list) * 255)[:, :3].astype("uint8").reshape(-1, 1, 3)


def create_opencv_colormap(src: np.ndarray, colormap_type: int) -> np.ndarray:
    """Create colormap image from matplotlib.

    Args:
        src : source image with specified shape
        colormap_type : colormap type

    Returns:
        array with colored values regarding specified colormap

    """
    return cv2.applyColorMap(src, colormap_type)


class ColorLabeler:
    """Class for labeling with color due to Colormap."""

    def __init__(
        self,
        map_type: Union[str, int] = cv2.COLORMAP_RAINBOW,
        map_creator: Callable = create_opencv_colormap,
        max_value: int = 256,
        normalized: bool = False,
    ):
        """Init.

        Args:
            map_type (int) : openCV colormap type
            max_value (int) : max value of color index
            normalized (boolean) : if True then all RGB values are in [0..1]

        """
        self.color_map_image_ = np.zeros((256, 30, 3), dtype="uint8")
        src = np.zeros((256, 1, 1), dtype="uint8")
        src[:, 0, 0] = range(0, 256)
        self.color_map_image_ = map_creator(src, map_type)
        self.mapsize = max_value
        if normalized:
            self.color_map_image_ = self.color_map_image_.astype("float32") / 255

    def label_to_color(self, label):
        """Convert label to color."""
        return tuple(
            int(val)
            for val in self.color_map_image_[int((label * 255) // self.mapsize) % 255, 0, :]
        )

    def label_to_color_norm(self, value):
        """Convert label to cv2 color."""
        return [int(val) for val in self.color_map_image_[int(value * 255), 0, :]]

    def label_to_color_id(self, color_idx):
        """Convert label to color."""
        return self.color_map_image_[color_idx, 0, :]

    def get_maxcolor(self):
        """Get maxcolor."""
        return self.mapsize

    def get_colormap(self):
        """Get colormap."""
        return self.color_map_image_

    def __call__(self, label):
        """Call for converting label to color."""
        return self.label_to_color(label)

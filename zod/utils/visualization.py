from typing import Tuple

import cv2
import numpy as np

from zod.constants import CAMERAS
from zod.utils.objects import Box3D
from zod.utils.zod_dataclasses import Calibration


def render_3d_box(
    image: np.ndarray,
    box3d: Box3D,
    calib: Calibration,
    color: Tuple[int, int, int] = (255, 0, 0),
    line_thickness=2,
) -> None:
    """Render a 3d box on the image."""

    assert box3d.frame in CAMERAS, "Only support rendering 3d boxes in camera frames."

    points = box3d.project_into_camera(calib).astype(np.int32)
    center_im = points[0]
    corners_im = points[1:]

    # draw the center point
    cv2.circle(image, center_im, 5, color, -1)

    # draw the 3d bounding box
    cv2.line(
        image,
        (corners_im[0][0], corners_im[0][1]),
        (corners_im[1][0], corners_im[1][1]),
        color,
        line_thickness,
    )
    cv2.line(
        image,
        (corners_im[1][0], corners_im[1][1]),
        (corners_im[2][0], corners_im[2][1]),
        color,
        line_thickness,
    )
    cv2.line(
        image,
        (corners_im[2][0], corners_im[2][1]),
        (corners_im[3][0], corners_im[3][1]),
        color,
        line_thickness,
    )
    cv2.line(
        image,
        (corners_im[3][0], corners_im[3][1]),
        (corners_im[0][0], corners_im[0][1]),
        color,
        line_thickness,
    )
    cv2.line(
        image,
        (corners_im[4][0], corners_im[4][1]),
        (corners_im[5][0], corners_im[5][1]),
        color,
        line_thickness,
    )
    cv2.line(
        image,
        (corners_im[5][0], corners_im[5][1]),
        (corners_im[6][0], corners_im[6][1]),
        color,
        line_thickness,
    )
    cv2.line(
        image,
        (corners_im[6][0], corners_im[6][1]),
        (corners_im[7][0], corners_im[7][1]),
        color,
        line_thickness,
    )
    cv2.line(
        image,
        (corners_im[7][0], corners_im[7][1]),
        (corners_im[4][0], corners_im[4][1]),
        color,
        line_thickness,
    )
    cv2.line(
        image,
        (corners_im[0][0], corners_im[0][1]),
        (corners_im[4][0], corners_im[4][1]),
        color,
        line_thickness,
    )
    cv2.line(
        image,
        (corners_im[1][0], corners_im[1][1]),
        (corners_im[5][0], corners_im[5][1]),
        color,
        line_thickness,
    )
    cv2.line(
        image,
        (corners_im[2][0], corners_im[2][1]),
        (corners_im[6][0], corners_im[6][1]),
        color,
        line_thickness,
    )
    cv2.line(
        image,
        (corners_im[3][0], corners_im[3][1]),
        (corners_im[7][0], corners_im[7][1]),
        color,
        line_thickness,
    )

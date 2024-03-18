from typing import List

import numpy as np


def quaternion_to_euler(qx: float, qy: float, qz: float, qw: float) -> List[float]:
    """
    Convert a quaternion into euler angles (roll, pitch, yaw).

    Parameters:
    qx (float): The x component of the quaternion.
    qy (float): The y component of the quaternion.
    qz (float): The z component of the quaternion.
    qw (float): The w component of the quaternion.

    Returns:
    list: The roll, pitch, and yaw euler angles in radians.
    """
    t0 = +2.0 * (qw * qx + qy * qz)
    t1 = +1.0 - 2.0 * (qx * qx + qy * qy)
    roll_x = np.arctan2(t0, t1)

    t2 = +2.0 * (qw * qy - qz * qx)
    t2 = np.clip(t2, a_min=-1.0, a_max=1.0)
    pitch_y = np.arcsin(t2)

    t3 = +2.0 * (qw * qz + qx * qy)
    t4 = +1.0 - 2.0 * (qy * qy + qz * qz)
    yaw_z = np.arctan2(t3, t4)

    return [roll_x, pitch_y, yaw_z]


def normalize_bbox(bbox, image_width=3848, image_height=2168):
    """
    Normalize the bounding box coordinates.

    Parameters:
    bbox (list or tuple): The bounding box in the format [x, y, width, height]
    image_width (int): Width of the image
    image_height (int): Height of the image

    Returns:
    list: Normalized bounding box [x_min, y_min, width, height]
    """
    x, y, width, height = bbox

    x_min = x / image_width
    y_min = y / image_height
    norm_width = width / image_width
    norm_height = height / image_height

    return [x_min, y_min, norm_width, norm_height]

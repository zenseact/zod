"""Geometry utility functions."""
from typing import Union
import numpy as np
from zod.constants import CAMERAS, EGO, LIDARS

from zod.utils.zod_dataclasses import Calibration


def project_3d_to_2d_kannala(
    points_3d: np.ndarray, camera_matrix: np.ndarray, distorion_coefs: np.ndarray
) -> np.ndarray:
    """Project 3d points to 2d using Kannala-Brandt model.

    Args:
        points_3d: 3d points in shape (N, 3). These are assumed to be in the
            camera frame to which they are beeing projected into.
        camera_matrix: Camera matrix in shape (3, 4).
        distorion_coefs: Distortion coefficients in shape (4,).

    Returns:
        2d points in shape (N, 2).
    """
    homogenous_points_3d = np.concatenate((points_3d, np.ones((points_3d.shape[0], 1))), axis=1)
    norm_data = np.linalg.norm(homogenous_points_3d[:, :2], axis=1)
    radial = np.arctan2(norm_data, homogenous_points_3d[:, 2])
    radial2 = radial**2
    radial4 = radial2**2
    radial6 = radial4 * radial2
    radial8 = radial4**2
    distortion_angle = radial * (
        1
        + distorion_coefs[0] * radial2
        + distorion_coefs[1] * radial4
        + distorion_coefs[2] * radial6
        + distorion_coefs[3] * radial8
    )
    u_dist = distortion_angle * homogenous_points_3d[:, 0] / norm_data
    v_dist = distortion_angle * homogenous_points_3d[:, 1] / norm_data
    pos_u = camera_matrix[0, 0] * u_dist + camera_matrix[0, 2]
    pos_v = camera_matrix[1, 1] * v_dist + camera_matrix[1, 2]

    return np.stack((pos_u, pos_v), axis=-1)


def unproject_2d_to_3d_kannala(
    points_2d: np.ndarray,
    camera_matrix: np.ndarray,
    undistorion_coefs: np.ndarray,
    depth: Union[np.array, float],
) -> np.ndarray:
    """Unproject 2d points to 3d using Kannala-Brandt model.

    Args:
        points_2d: 2d points in shape (N, 2).
        camera_matrix: Camera matrix in shape (3, 4).
        undistorion_coefs: Unistortion coefficients in shape (4,).
        depth: Depth of the points in the camera frame.

    Returns:
        3d points in shape (N, 3).
    """
    N = points_2d.shape[0]
    assert np.isscalar(depth) or depth.shape == (N,)

    out = np.ones((N, 3))
    pixel_x, pixel_y = points_2d[:, 0], points_2d[:, 1]
    focal_x, focal_y = camera_matrix[0, 0], camera_matrix[1, 1]
    principal_x, principal_y = camera_matrix[0, 2], camera_matrix[1, 2]

    out[:, 0] = (pixel_x - principal_x) / focal_x
    out[:, 1] = (pixel_y - principal_y) / focal_y
    rho = np.linalg.norm(out[:, :2], axis=1)
    phi = np.arctan2(out[:, 1], out[:, 0])
    theta = rho * (
        1
        + undistorion_coefs[0] * rho**2
        + undistorion_coefs[1] * rho**4
        + undistorion_coefs[2] * rho**6
        + undistorion_coefs[3] * rho**8,
    )
    out[:, 0] = np.sin(theta) * np.cos(phi) * depth
    out[:, 1] = np.sin(theta) * np.sin(phi) * depth
    out[:, 2] = np.cos(theta) * depth

    return out


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Transform points from one frame to another.

    Args:
        points: Points in shape (N, 3).
        transform: Transform matrix in shape (4, 4).
    """
    points_homogenous = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
    transformed_points = np.matmul(points_homogenous, transform.T)
    # Normalize the points
    transformed_points = transformed_points[:, :3] / transformed_points[:, 3:]
    # Remove the last dimension
    transformed_points = transformed_points[:, :3]
    return transformed_points


def transform_points(
    points: np.ndarray, from_frame: str, to_frame: str, calibration: Calibration
) -> np.ndarray:
    """Transform points from one frame to another.

    Args:
        points: Points in shape (N, 3).
        from_frame: Name of the frame the points are in.
        to_frame: Name of the frame to transform the points to.
        calibration: Calibration object.

    Returns:
        Transformed points in shape (N, 3).
    """
    assert to_frame in (*LIDARS, *CAMERAS, EGO)
    assert from_frame in (*LIDARS, *CAMERAS, EGO)
    if from_frame == to_frame:
        return points

    if from_frame == EGO:
        from_transform = np.eye(4)
    elif from_frame in LIDARS:
        from_transform = calibration.lidars[from_frame].extrinsics.transform
    elif from_frame in CAMERAS:
        from_transform = calibration.cameras[from_frame].extrinsics.transform

    if to_frame == EGO:
        to_transform = np.eye(4)
    elif to_frame in LIDARS:
        to_transform = np.linalg.pinv(calibration.lidars[to_frame].extrinsics.transform)
    elif to_frame in CAMERAS:
        to_transform = np.linalg.pinv(calibration.cameras[to_frame].extrinsics.transform)

    return _transform_points(points, to_transform @ from_transform)

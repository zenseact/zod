"""Module to perform OxTS extraction and visualize GPS track projection on image plane."""

import cv2
import numpy as np

from zod.constants import Camera
from zod.data_classes.calibration import Calibration
from zod.data_classes.ego_motion import EgoMotion
from zod.utils.geometry import get_points_in_camera_fov, project_3d_to_2d_kannala, transform_points


def visualize_oxts_on_image(oxts: EgoMotion, key_timestamp, calibs: Calibration, image, camera=Camera.FRONT):
    """Visualize oxts track on image plane."""

    # get pose at key frame
    current_pose = oxts.get_poses(key_timestamp)
    all_poses = oxts.poses

    # transform all poses to the current pose
    transformed_poses = np.linalg.pinv(current_pose) @ all_poses

    # get the positions from the transformed poses
    points = transformed_poses[:, :3, -1]

    # let's filter out the points that are behind the camera
    points = points[points[:, 0] > 0]

    # transform point to camera coordinate system
    T_inv = np.linalg.pinv(calibs.get_extrinsics(camera).transform)
    camerapoints = transform_points(points[:, :3], T_inv)
    print(f"Number of points: {points.shape[0]}")

    # filter points that are not in the camera field of view
    points_in_fov, _ = get_points_in_camera_fov(
        calibs.cameras[camera].field_of_view, camerapoints, horizontal_only=True
    )
    print(f"Number of points in fov: {len(points_in_fov)}")

    # project points to image plane
    xy_array = project_3d_to_2d_kannala(
        points_in_fov,
        calibs.cameras[camera].intrinsics[..., :3],
        calibs.cameras[camera].distortion,
    )

    for i in range(xy_array.shape[0]):
        cv2.circle(image, (int(xy_array[i, 0]), int(xy_array[i, 1])), 3, (255, 0, 0), -1)

    return image

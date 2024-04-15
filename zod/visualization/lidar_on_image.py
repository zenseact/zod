"""Module to perform Lidar extraction and visualization of projections on image plane."""

from typing import Tuple, Union

import cv2
import numpy as np

from zod.constants import Camera, Lidar
from zod.data_classes.calibration import Calibration
from zod.data_classes.geometry import Pose
from zod.data_classes.sensor import LidarData
from zod.utils.geometry import get_points_in_camera_fov, project_3d_to_2d_kannala, transform_points
from zod.visualization.colorlabeler import ColorLabeler, create_matplotlib_colormap


def get_3d_transform_camera_lidar(
    calib: Calibration,
    lidar: Lidar = Lidar.VELODYNE,
    camera: Camera = Camera.FRONT,
) -> Pose:
    """Get 3D transformation between lidar and camera."""
    t_refframe_to_frame = calib.lidars[lidar].extrinsics
    t_refframe_from_frame = calib.cameras[camera].extrinsics

    t_from_frame_refframe = t_refframe_from_frame.inverse
    t_from_frame_to_frame = Pose(t_from_frame_refframe.transform @ t_refframe_to_frame.transform)

    return t_from_frame_to_frame


def project_lidar_to_image(
    lidar_data: LidarData,
    calib: Calibration,
    lidar: Lidar = Lidar.VELODYNE,
    camera: Camera = Camera.FRONT,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project lidar pointcloud to camera (also returns mask)."""
    t_lidar_to_camera = get_3d_transform_camera_lidar(calib, lidar, camera)

    camera_data = transform_points(lidar_data.points, t_lidar_to_camera.transform)
    positive_depth = camera_data[:, 2] > 0
    camera_data = camera_data[positive_depth]
    if not camera_data.any():
        return camera_data, positive_depth

    camera_data, mask = get_points_in_camera_fov(calib.cameras[camera].field_of_view, camera_data)
    xy_array = project_3d_to_2d_kannala(
        camera_data,
        calib.cameras[camera].intrinsics[..., :3],
        calib.cameras[camera].distortion,
    )
    xyd_array = np.concatenate([xy_array, camera_data[:, 2:3]], axis=1)
    final_mask = np.zeros_like(positive_depth)
    final_mask[positive_depth] = mask
    return xyd_array, final_mask


def draw_projections_as_points(image: np.ndarray, points: np.ndarray, clip_to: float = None) -> np.ndarray:
    """Draw projected points from pointcloud to image plane as colored points.

    Args:
        image : image to draw projected points
        points: projected points from lidar
        clip_to : distance for clipping

    Returns:
        image : image with projected lidar points as colored points

    """
    color_labeler = ColorLabeler(map_type="coolwarm", map_creator=create_matplotlib_colormap)
    if points.shape[0]:
        normed_depths = points[:, 2] - points[:, 2].min()
        if clip_to:
            normed_depths = np.clip(normed_depths / clip_to, 0, 1)
        else:
            normed_depths = normed_depths / normed_depths.max()
        colors = [color_labeler.label_to_color_norm(d) for d in normed_depths]
        coords = points[:, :2].astype("int32")
        image[coords[:, 1], coords[:, 0]] = colors
    return image


def draw_projection_as_jet_circles(
    image: np.ndarray, points: np.ndarray, radius: int, clip_to: float = None
) -> np.ndarray:
    """Draw projected points from pointcloud to image plane as jet colored circles.

    Args:
        image : image to draw projected points
        points: projected points from lidar
        radius : radius of circle to be drawed
        clip_to : distance for clipping

    Returns:
        image : image with projected lidar points as colored circles with specified radius

    """
    color_labeler = ColorLabeler(map_type="coolwarm", map_creator=create_matplotlib_colormap)
    if points.shape[0]:
        normed_depths = points[:, 2] - points[:, 2].min()
        if clip_to:
            normed_depths = np.clip(normed_depths / clip_to, 0, 1)
        else:
            normed_depths = normed_depths / normed_depths.max()
        colors = [color_labeler.label_to_color_norm(d) for d in normed_depths]
        for point, color in zip(points[:, :2].astype("int32"), colors):
            center = tuple(point)
            cv2.circle(image, center, radius, color, -1)
    return image


def visualize_lidar_on_image(lidar_data: LidarData, calib: Calibration, image: np.ndarray):
    """Visualize GPS track on image."""
    xyd, _ = project_lidar_to_image(lidar_data, calib)
    image = draw_projection_as_jet_circles(image, xyd, radius=2)
    return image

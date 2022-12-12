"""Module to perform Lidar extraction and visualization of projections on image plane."""

import cv2
import numpy as np
from calibration import CameraInfo, get_3d_transform_camera_lidar, rigid_transform_3d
from colorlabeler import ColorLabeler, create_matplotlib_colormap

from constants import FOV


def project_lidar_to_image(xyz_data, calib: dict) -> np.ndarray:
    """Project lidar pointcloud to camera."""
    t_lidar_to_camera = get_3d_transform_camera_lidar(calib)

    camera_data = rigid_transform_3d(xyz_data, t_lidar_to_camera)
    positive_depth = camera_data[:, 2] > 0
    camera_data = camera_data[positive_depth]
    if not camera_data.any():
        return camera_data
    camera_data = get_points_in_camera_fov(calib[FOV][0], camera_data)

    camera_info = CameraInfo.create_instance_from_config(calib)
    xyd_array = camera_info.project(camera_data, True)
    return xyd_array


def get_points_in_camera_fov(horizontal_fov: int, camera_data: np.ndarray) -> np.ndarray:
    """Get points that are present in camera field of view.

    Args:
        horizontal_fov: horizontal camera field of view
        camera_data: data to filter inside the camera field of view

    Returns:
        points only visible in the camera

    """
    if horizontal_fov == 0:
        return camera_data
    angles = np.rad2deg(np.arctan2(camera_data[:, 0], camera_data[:, 2]))
    mask = np.logical_and(angles > -horizontal_fov / 2, angles < horizontal_fov / 2)
    return camera_data[mask.flatten()]


def draw_projections_as_points(
    image: np.array, points: np.array, clip_to: float = None
) -> np.array:
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
    image: np.array, points: np.array, radius: int, clip_to: float = None
) -> np.array:
    """Draw projected points from pointcloud to image plane as jet colored circles.

    Args:
        image : image to draw projected points
        points: projected points from lidar
        radius : radius of circle to be drawed
        clip_to : distance for clipping

    Returns:
        image : image with projected lidar points as colored circles with specified radius

    """
    color_labeler = ColorLabeler()
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


def visualize_lidar_on_image(lidar_data: np.ndarray, calib: dict, image: np.ndarray):
    """Visualize GPS track on image."""
    xyd = project_lidar_to_image(lidar_data, calib)
    image = draw_projection_as_jet_circles(image, xyd, radius=2)
    return image

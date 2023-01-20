"""Module to perform Lidar extraction and visualization of projections on image plane."""

import cv2
import numpy as np

from zod.constants import Camera, Lidar
from zod.utils.geometry import transform_points

# from calibration import CameraInfo, get_3d_transform_camera_lidar, rigid_transform_3d
from zod.visualization.colorlabeler import ColorLabeler, create_matplotlib_colormap
from zod.visualization.oxts_on_image import kannala_project
from zod.zod_dataclasses.calibration import Calibration
from zod.zod_dataclasses.geometry import Pose
from zod.zod_dataclasses.sensor import LidarData

# from constants import FOV


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
) -> np.ndarray:
    """Project lidar pointcloud to camera."""
    # TODO: use lidar param here
    t_lidar_to_camera = get_3d_transform_camera_lidar(calib)

    camera_data = transform_points(lidar_data.points, t_lidar_to_camera.transform)
    positive_depth = camera_data[:, 2] > 0
    camera_data = camera_data[positive_depth]
    if not camera_data.any():
        return camera_data
    camera_data = get_points_in_camera_fov(calib.cameras[camera].field_of_view, camera_data)

    xy_array = kannala_project(
        camera_data,
        calib.cameras[camera].intrinsics[..., :3],
        calib.cameras[camera].distortion,
    )
    xyd_array = np.concatenate([xy_array, camera_data[:, 2:3]], axis=1)
    return xyd_array


def get_points_in_camera_fov(fov: np.ndarray, camera_data: np.ndarray) -> np.ndarray:
    """Get points that are present in camera field of view.

    Args:
        fov: camera field of view
        camera_data: data to filter inside the camera field of view

    Returns:
        points only visible in the camera

    """
    horizontal_fov, vertical_fov = fov
    if horizontal_fov == 0:
        return camera_data
    angles = np.rad2deg(np.arctan2(camera_data[:, 0], camera_data[:, 2]))
    mask = np.logical_and(angles > -horizontal_fov / 2, angles < horizontal_fov / 2)
    return camera_data[mask.flatten()]


def draw_projections_as_points(
    image: np.ndarray, points: np.ndarray, clip_to: float = None
) -> np.ndarray:
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
    xyd = project_lidar_to_image(lidar_data, calib)
    image = draw_projection_as_jet_circles(image, xyd, radius=2)
    return image

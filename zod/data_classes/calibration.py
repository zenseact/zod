"""Calibration dataclasses."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from zod.constants import EGO, Camera, CoordinateFrame, Lidar
from zod.data_classes.geometry import Pose
from zod.utils.geometry import transform_points


@dataclass
class LidarCalibration:
    extrinsics: Pose  # lidar pose in the ego frame


@dataclass
class CameraCalibration:
    extrinsics: Pose  # 4x4 matrix describing the camera pose in the ego frame
    intrinsics: np.ndarray  # 3x3 matrix
    distortion: np.ndarray  # 4 vector
    undistortion: np.ndarray  # 4 vector
    image_dimensions: np.ndarray  # width, height
    field_of_view: np.ndarray  # horizontal, vertical (degrees)


@dataclass
class Calibration:
    lidars: Dict[Lidar, LidarCalibration]
    cameras: Dict[Camera, CameraCalibration]

    @classmethod
    def from_dict(cls, calib_dict: Dict[str, Any]) -> Calibration:
        lidars = {
            Lidar.VELODYNE: LidarCalibration(extrinsics=Pose(np.array(calib_dict["FC"]["lidar_extrinsics"]))),
        }
        cameras = {
            Camera.FRONT: CameraCalibration(
                extrinsics=Pose(np.array(calib_dict["FC"]["extrinsics"])),
                intrinsics=np.array(calib_dict["FC"]["intrinsics"]),
                distortion=np.array(calib_dict["FC"]["distortion"]),
                undistortion=np.array(calib_dict["FC"]["undistortion"]),
                image_dimensions=np.array(calib_dict["FC"]["image_dimensions"]),
                field_of_view=np.array(calib_dict["FC"]["field_of_view"]),
            ),
        }
        return cls(lidars=lidars, cameras=cameras)

    @classmethod
    def from_json_path(cls, json_path: str) -> Calibration:
        with open(json_path) as f:
            calib_dict = json.load(f)
        return cls.from_dict(calib_dict)

    def transform_points(
        self,
        points: np.ndarray,
        from_frame: CoordinateFrame,
        to_frame: CoordinateFrame,
    ) -> np.ndarray:
        """Transform points from one frame to another.

        Args:
            points: Points in shape (N, 3).
            from_frame: Name of the frame the points are in.
            to_frame: Name of the frame to transform the points to.

        Returns:
            Transformed points in shape (N, 3).
        """
        if from_frame == to_frame:
            return points
        from_transform = self.get_extrinsics(from_frame).transform
        to_transform = self.get_extrinsics(to_frame).transform
        return transform_points(points, to_transform @ from_transform)

    def get_extrinsics(self, frame: CoordinateFrame) -> Pose:
        if frame == EGO:
            return Pose.identity()
        elif isinstance(frame, Lidar):
            return self.lidars[frame].extrinsics
        elif isinstance(frame, Camera):
            return self.cameras[frame].extrinsics
        else:
            raise ValueError(f"Unknown frame {frame}")

"""ZOD dataclasses."""
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Union

import numpy as np
from dataclass_wizard import JSONSerializable
from pyquaternion import Quaternion

from zod.constants import CAMERA_FRONT, CAMERAS, EGO, LIDAR_VELODYNE, LIDARS
from zod.utils.geometry import transform_points


@dataclass
class Pose:
    """A general class describing some pose."""

    transform: np.ndarray

    @property
    def translation(self) -> np.ndarray:
        """Return the translation (array)."""
        return self.transform[:3, 3]

    @property
    def rotation(self) -> Quaternion:
        """Return the rotation as a quaternion."""
        return Quaternion(matrix=self.rotation_matrix)

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Return the rotation matrix."""
        return self.transform[:3, :3]

    @property
    def inverse(self) -> "Pose":
        """Return the inverse of the pose."""
        return Pose(np.linalg.inv(self.transform))

    @classmethod
    def from_translation_rotation(
        cls, translation: np.ndarray, rotation_matrix: np.ndarray
    ) -> "Pose":
        """Create a pose from a translation and a rotation."""
        transform = np.eye(4, 4)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = translation
        return cls(transform)


@dataclass
class LidarData:
    """A class describing the lidar data."""

    points: np.ndarray  # (N, 3) float32
    timestamps: np.ndarray  # (N,) float64 epoch timestamp pointwise
    intensity: np.ndarray  # (N,) uint8
    diode_idx: np.ndarray  # (N,) uint8
    core_timestamp: np.float64  # core epoch timestamp

    def copy(self) -> "LidarData":
        """Return a copy of the lidar data."""
        return LidarData(
            points=self.points.copy(),
            timestamps=self.timestamps.copy(),
            intensity=self.intensity.copy(),
            diode_idx=self.diode_idx.copy(),
            core_timestamp=self.core_timestamp,
        )

    @classmethod
    def empty(cls) -> "LidarData":
        """Create an empty lidar data object."""
        return cls(
            points=np.empty((0, 3), dtype=np.float32),
            timestamps=np.empty(0, dtype=np.float64),
            intensity=np.empty(0, dtype=np.uint8),
            diode_idx=np.empty(0, dtype=np.uint8),
            core_timestamp=np.float64(0),
        )

    @classmethod
    def from_npy(cls, path: str) -> "LidarData":
        """Load lidar data from a .npy file."""
        data = np.load(path)
        core_datetime = parse_datetime_from_filename(path)
        core_timestamp = np.float64(core_datetime.timestamp())

        return cls(
            points=np.vstack((data["x"], data["y"], data["z"])).T,
            timestamps=core_timestamp + data["timestamp"] / 1e6,
            intensity=data["intensity"],
            diode_idx=data["diode_index"],
            core_timestamp=core_timestamp,
        )

    def to_npy(self, path: str) -> None:
        """Save lidar data to a .npy file in the same format as is used for loading."""
        data = np.empty(
            len(self.points),
            dtype=[
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
                ("timestamp", np.int64),
                ("intensity", np.uint8),
                ("diode_index", np.uint8),
            ],
        )
        data["x"] = self.points[:, 0]
        data["y"] = self.points[:, 1]
        data["z"] = self.points[:, 2]
        data["timestamp"] = (self.timestamps - self.core_timestamp) * 1e6
        data["intensity"] = self.intensity
        data["diode_index"] = self.diode_idx
        np.save(path, data)

    def transform(self, pose: Union[np.ndarray, Pose]) -> None:
        """Transform the lidar data to a new pose.

        Args:
            pose: The new pose to transform the lidar data to.
        """
        if isinstance(pose, Pose):
            pose = pose.transform
        rotations = pose[..., :3, :3]
        translations = pose[..., :3, 3]
        self.points = (
            self.points[..., None, :] @ rotations.swapaxes(-2, -1) + translations[..., None, :]
        )

    def append(self, other: "LidarData") -> "LidarData":
        """Append another LidarData object to this one.

        Args:
            other: The other LidarData object to append.

        Returns:
            The appended LidarData object.
        """
        self.points = np.vstack((self.points, other.points))
        self.timestamps = np.hstack((self.timestamps, other.timestamps))
        self.intensity = np.hstack((self.intensity, other.intensity))
        self.diode_idx = np.hstack((self.diode_idx, other.diode_idx))
        # Core timestamp is the weighted average
        self.core_timestamp = (
            self.core_timestamp * len(self.timestamps)
            + other.core_timestamp * len(other.timestamps)
        ) / (len(self.timestamps) + len(other.timestamps))
        return self


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
    lidars: Dict[str, LidarCalibration]
    cameras: Dict[str, CameraCalibration]

    @classmethod
    def from_dict(cls, calib_dict: Dict[str, Any]) -> "Calibration":
        lidars = {
            LIDAR_VELODYNE: LidarCalibration(
                extrinsics=Pose(np.array(calib_dict["FC"]["lidar_extrinsics"]))
            ),
        }
        cameras = {
            CAMERA_FRONT: CameraCalibration(
                extrinsics=Pose(np.array(calib_dict["FC"]["extrinsics"])),
                intrinsics=np.array(calib_dict["FC"]["intrinsics"]),
                distortion=np.array(calib_dict["FC"]["distortion"]),
                undistortion=np.array(calib_dict["FC"]["undistortion"]),
                image_dimensions=np.array(calib_dict["FC"]["image_dimensions"]),
                field_of_view=np.array(calib_dict["FC"]["field_of_view"]),
            ),
        }
        return cls(lidars=lidars, cameras=cameras)

    def transform_points(self, points: np.ndarray, from_frame: str, to_frame: str) -> np.ndarray:
        """Transform points from one frame to another.

        Args:
            points: Points in shape (N, 3).
            from_frame: Name of the frame the points are in.
            to_frame: Name of the frame to transform the points to.

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
            from_transform = self.lidars[from_frame].extrinsics.transform
        elif from_frame in CAMERAS:
            from_transform = self.cameras[from_frame].extrinsics.transform

        if to_frame == EGO:
            to_transform = np.eye(4)
        elif to_frame in LIDARS:
            to_transform = np.linalg.pinv(self.lidars[to_frame].extrinsics.transform)
        elif to_frame in CAMERAS:
            to_transform = np.linalg.pinv(self.cameras[to_frame].extrinsics.transform)

        return transform_points(points, to_transform @ from_transform)


@dataclass
class MetaData:
    """A class describing the metadata of a frame."""

    frame_id: str
    timestamp: datetime
    country_code: str
    scraped_weather: str
    collection_car: str
    road_type: str
    road_condition: str
    time_of_day: str
    num_lane_instances: int
    num_vehicles: int
    num_vulnerable_vehicles: int
    num_pedestrians: int
    num_traffic_lights: int
    num_traffic_signs: int
    longitude: float
    latitude: float
    solar_angle_elevation: float

    @classmethod
    def from_dict(cls, meta_dict: Dict[str, Any]):
        """Create a MetaData object from a dictionary."""
        return cls(**meta_dict)


@dataclass
class SensorFrame(JSONSerializable):
    """Class to store sensor information."""

    filepath: str
    timestamp: datetime


@dataclass
class CameraFrame(SensorFrame):
    """Class to store sensor information."""

    height: int = 2168
    width: int = 3848

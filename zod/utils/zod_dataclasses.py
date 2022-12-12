"""ZOD dataclasses."""
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Union

import h5py
import numpy as np
from dataclass_wizard import JSONSerializable
from pyquaternion import Quaternion

from zod.constants import CAMERA_FRONT, LIDAR_VELODYNE
from zod.utils.utils import parse_timestamp_from_filename


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
class OXTSData:
    acceleration_x: np.ndarray
    acceleration_y: np.ndarray
    acceleration_z: np.ndarray
    angular_rate_x: np.ndarray
    angular_rate_y: np.ndarray
    angular_rate_z: np.ndarray
    ecef_x: np.ndarray
    ecef_y: np.ndarray
    ecef_z: np.ndarray
    heading: np.ndarray
    leap_seconds: np.ndarray
    pitch: np.ndarray
    pos_alt: np.ndarray
    pos_lat: np.ndarray
    pos_lon: np.ndarray
    roll: np.ndarray
    std_dev_pos_east: np.ndarray
    std_dev_pos_north: np.ndarray
    time_gps: np.ndarray
    traveled: np.ndarray
    vel_down: np.ndarray
    vel_forward: np.ndarray
    vel_lateral: np.ndarray

    def get_ego_pose(self, timestamp: datetime) -> Pose:
        return Pose(np.eye(4, 4))

    @classmethod
    def from_hdf5(cls, file: h5py.Group) -> "OXTSData":
        return cls(
            acceleration_x=np.array(file["accelerationX"]),
            acceleration_y=np.array(file["accelerationY"]),
            acceleration_z=np.array(file["accelerationZ"]),
            angular_rate_x=np.array(file["angularRateX"]),
            angular_rate_y=np.array(file["angularRateY"]),
            angular_rate_z=np.array(file["angularRateZ"]),
            ecef_x=np.array(file["ecef_x"]),
            ecef_y=np.array(file["ecef_y"]),
            ecef_z=np.array(file["ecef_z"]),
            heading=np.array(file["heading"]),
            leap_seconds=np.array(file["leapSeconds"]),
            pitch=np.array(file["pitch"]),
            pos_alt=np.array(file["posAlt"]),
            pos_lat=np.array(file["posLat"]),
            pos_lon=np.array(file["posLon"]),
            roll=np.array(file["roll"]),
            std_dev_pos_east=np.array(file["stdDevPosEast"]),
            std_dev_pos_north=np.array(file["stdDevPosNorth"]),
            time_gps=np.array(file["time_gps"]),
            traveled=np.array(file["traveled"]),
            vel_down=np.array(file["velDown"]),
            vel_forward=np.array(file["velForward"]),
            vel_lateral=np.array(file["velLateral"]),
        )


@dataclass
class LidarData:
    """A class describing the lidar data."""

    points: np.ndarray  # (N, 3) float32
    timestamps: np.ndarray  # (N,) int64 epoch time in microseconds
    intensity: np.ndarray  # (N,) uint8
    diode_idx: np.ndarray  # (N,) uint8

    @classmethod
    def from_npy(cls, path: str) -> "LidarData":
        """Load lidar data from a .npy file."""
        data = np.load(path)
        core_timestamp = parse_timestamp_from_filename(path)
        core_timestamp_epoch_us = np.int64(core_timestamp.timestamp() * 1e6)

        return cls(
            points=np.vstack((data["x"], data["y"], data["z"])).T,
            timestamps=core_timestamp_epoch_us + data["timestamp"],
            intensity=data["intensity"],
            diode_idx=data["diode_index"],
        )


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
    field_of_view: np.ndarray  # vertical, horizontal (degrees)


@dataclass
class Calibration:
    lidars: Dict[str, LidarCalibration]
    cameras: Dict[str, CameraCalibration]

    @classmethod
    def from_dict(cls, calib_dict: Dict[str, Any]):
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


@dataclass
class FrameInformation(JSONSerializable):
    """Class to store frame information."""

    frame_id: str
    timestamp: datetime

    traffic_sign_annotation_path: Union[str, None]
    ego_road_annotation_path: Union[str, None]
    object_detection_annotation_path: str
    lane_markings_annotation_path: str
    road_condition_annotation_path: str

    lidar_frame: Dict[str, SensorFrame]
    # these are chronologicaly ordered, i.e, first entry in this list is
    # the furthest away from the core lidar_frame
    previous_lidar_frames: Dict[str, List[SensorFrame]]
    # these are chronologicaly ordered, i.e, first entry in this list is
    # the closest to the core lidar_frame
    future_lidar_frames: Dict[str, List[SensorFrame]]
    # TODO: list available cameras, maybe with a nice error message
    camera_frame: Dict[str, CameraFrame]

    oxts_path: str
    calibration_path: str

    metadata_path: str

    def convert_paths_to_absolute(self, root_path: str):
        self.traffic_sign_annotation_path = (
            os.path.join(root_path, self.traffic_sign_annotation_path)
            if self.traffic_sign_annotation_path
            else None
        )
        self.ego_road_annotation_path = (
            os.path.join(root_path, self.ego_road_annotation_path)
            if self.ego_road_annotation_path
            else None
        )
        self.object_detection_annotation_path = os.path.join(
            root_path, self.object_detection_annotation_path
        )
        self.lane_markings_annotation_path = os.path.join(
            root_path, self.lane_markings_annotation_path
        )
        self.road_condition_annotation_path = os.path.join(
            root_path, self.road_condition_annotation_path
        )
        self.oxts_path = os.path.join(root_path, self.oxts_path)
        self.calibration_path = os.path.join(root_path, self.calibration_path)
        self.metadata_path = os.path.join(root_path, self.metadata_path)
        for sensor_frame in self.lidar_frame.values():
            sensor_frame.filepath = os.path.join(root_path, sensor_frame.filepath)
        for lidar_frames in self.previous_lidar_frames.values():
            for sensor_frame in lidar_frames:
                sensor_frame.filepath = os.path.join(root_path, sensor_frame.filepath)
        for lidar_frames in self.future_lidar_frames.values():
            for sensor_frame in lidar_frames:
                sensor_frame.filepath = os.path.join(root_path, sensor_frame.filepath)
        for sensor_frame in self.camera_frame.values():
            sensor_frame.filepath = os.path.join(root_path, sensor_frame.filepath)

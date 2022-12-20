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

    def __len__(self) -> int:
        return len(self.time_gps)

    def array_mul(self, array):
        return OXTSData(
            acceleration_x=self.acceleration_x * array,
            acceleration_y=self.acceleration_y * array,
            acceleration_z=self.acceleration_z * array,
            angular_rate_x=self.angular_rate_x * array,
            angular_rate_y=self.angular_rate_y * array,
            angular_rate_z=self.angular_rate_z * array,
            ecef_x=self.ecef_x * array,
            ecef_y=self.ecef_y * array,
            ecef_z=self.ecef_z * array,
            heading=self.heading * array,
            leap_seconds=self.leap_seconds * array,
            pitch=self.pitch * array,
            pos_alt=self.pos_alt * array,
            pos_lat=self.pos_lat * array,
            pos_lon=self.pos_lon * array,
            roll=self.roll * array,
            std_dev_pos_east=self.std_dev_pos_east * array,
            std_dev_pos_north=self.std_dev_pos_north * array,
            time_gps=self.time_gps * array,
            traveled=self.traveled * array,
            vel_down=self.vel_down * array,
            vel_forward=self.vel_forward * array,
            vel_lateral=self.vel_lateral * array,
        )

    def __mul__(self, other: Union[float, np.ndarray]) -> "OXTSData":
        return OXTSData(
            acceleration_x=self.acceleration_x * other,
            acceleration_y=self.acceleration_y * other,
            acceleration_z=self.acceleration_z * other,
            angular_rate_x=self.angular_rate_x * other,
            angular_rate_y=self.angular_rate_y * other,
            angular_rate_z=self.angular_rate_z * other,
            ecef_x=self.ecef_x * other,
            ecef_y=self.ecef_y * other,
            ecef_z=self.ecef_z * other,
            heading=self.heading * other,
            leap_seconds=self.leap_seconds * other,
            pitch=self.pitch * other,
            pos_alt=self.pos_alt * other,
            pos_lat=self.pos_lat * other,
            pos_lon=self.pos_lon * other,
            roll=self.roll * other,
            std_dev_pos_east=self.std_dev_pos_east * other,
            std_dev_pos_north=self.std_dev_pos_north * other,
            time_gps=self.time_gps * other,
            traveled=self.traveled * other,
            vel_down=self.vel_down * other,
            vel_forward=self.vel_forward * other,
            vel_lateral=self.vel_lateral * other,
        )

    def __rmul__(self, other: float) -> "OXTSData":
        return self.__mul__(other)

    def __add__(self, other: "OXTSData") -> "OXTSData":
        return OXTSData(
            acceleration_x=self.acceleration_x + other.acceleration_x,
            acceleration_y=self.acceleration_y + other.acceleration_y,
            acceleration_z=self.acceleration_z + other.acceleration_z,
            angular_rate_x=self.angular_rate_x + other.angular_rate_x,
            angular_rate_y=self.angular_rate_y + other.angular_rate_y,
            angular_rate_z=self.angular_rate_z + other.angular_rate_z,
            ecef_x=self.ecef_x + other.ecef_x,
            ecef_y=self.ecef_y + other.ecef_y,
            ecef_z=self.ecef_z + other.ecef_z,
            heading=self.heading + other.heading,
            leap_seconds=self.leap_seconds + other.leap_seconds,
            pitch=self.pitch + other.pitch,
            pos_alt=self.pos_alt + other.pos_alt,
            pos_lat=self.pos_lat + other.pos_lat,
            pos_lon=self.pos_lon + other.pos_lon,
            roll=self.roll + other.roll,
            std_dev_pos_east=self.std_dev_pos_east + other.std_dev_pos_east,
            std_dev_pos_north=self.std_dev_pos_north + other.std_dev_pos_north,
            time_gps=self.time_gps + other.time_gps,
            traveled=self.traveled + other.traveled,
            vel_down=self.vel_down + other.vel_down,
            vel_forward=self.vel_forward + other.vel_forward,
            vel_lateral=self.vel_lateral + other.vel_lateral,
        )

    def __sub__(self, other: "OXTSData") -> "OXTSData":
        return OXTSData(
            acceleration_x=self.acceleration_x - other.acceleration_x,
            acceleration_y=self.acceleration_y - other.acceleration_y,
            acceleration_z=self.acceleration_z - other.acceleration_z,
            angular_rate_x=self.angular_rate_x - other.angular_rate_x,
            angular_rate_y=self.angular_rate_y - other.angular_rate_y,
            angular_rate_z=self.angular_rate_z - other.angular_rate_z,
            ecef_x=self.ecef_x - other.ecef_x,
            ecef_y=self.ecef_y - other.ecef_y,
            ecef_z=self.ecef_z - other.ecef_z,
            heading=self.heading - other.heading,
            leap_seconds=self.leap_seconds - other.leap_seconds,
            pitch=self.pitch - other.pitch,
            pos_alt=self.pos_alt - other.pos_alt,
            pos_lat=self.pos_lat - other.pos_lat,
            pos_lon=self.pos_lon - other.pos_lon,
            roll=self.roll - other.roll,
            std_dev_pos_east=self.std_dev_pos_east - other.std_dev_pos_east,
            std_dev_pos_north=self.std_dev_pos_north - other.std_dev_pos_north,
            time_gps=self.time_gps - other.time_gps,
            traveled=self.traveled - other.traveled,
            vel_down=self.vel_down - other.vel_down,
            vel_forward=self.vel_forward - other.vel_forward,
            vel_lateral=self.vel_lateral - other.vel_lateral,
        )

    def __truediv__(self, other: float) -> "OXTSData":
        return OXTSData(
            acceleration_x=self.acceleration_x / other,
            acceleration_y=self.acceleration_y / other,
            acceleration_z=self.acceleration_z / other,
            angular_rate_x=self.angular_rate_x / other,
            angular_rate_y=self.angular_rate_y / other,
            angular_rate_z=self.angular_rate_z / other,
            ecef_x=self.ecef_x / other,
            ecef_y=self.ecef_y / other,
            ecef_z=self.ecef_z / other,
            heading=self.heading / other,
            leap_seconds=self.leap_seconds / other,
            pitch=self.pitch / other,
            pos_alt=self.pos_alt / other,
            pos_lat=self.pos_lat / other,
            pos_lon=self.pos_lon / other,
            roll=self.roll / other,
            std_dev_pos_east=self.std_dev_pos_east / other,
            std_dev_pos_north=self.std_dev_pos_north / other,
            time_gps=self.time_gps / other,
            traveled=self.traveled / other,
            vel_down=self.vel_down / other,
            vel_forward=self.vel_forward / other,
            vel_lateral=self.vel_lateral / other,
        )

    def append(self, other: "OXTSData") -> "OXTSData":
        self.acceleration_x = np.append(self.acceleration_x, other.acceleration_x)
        self.acceleration_y = np.append(self.acceleration_y, other.acceleration_y)
        self.acceleration_z = np.append(self.acceleration_z, other.acceleration_z)
        self.angular_rate_x = np.append(self.angular_rate_x, other.angular_rate_x)
        self.angular_rate_y = np.append(self.angular_rate_y, other.angular_rate_y)
        self.angular_rate_z = np.append(self.angular_rate_z, other.angular_rate_z)
        self.ecef_x = np.append(self.ecef_x, other.ecef_x)
        self.ecef_y = np.append(self.ecef_y, other.ecef_y)
        self.ecef_z = np.append(self.ecef_z, other.ecef_z)
        self.heading = np.append(self.heading, other.heading)
        self.leap_seconds = np.append(self.leap_seconds, other.leap_seconds)
        self.pitch = np.append(self.pitch, other.pitch)
        self.pos_alt = np.append(self.pos_alt, other.pos_alt)
        self.pos_lat = np.append(self.pos_lat, other.pos_lat)
        self.pos_lon = np.append(self.pos_lon, other.pos_lon)
        self.roll = np.append(self.roll, other.roll)
        self.std_dev_pos_east = np.append(self.std_dev_pos_east, other.std_dev_pos_east)
        self.std_dev_pos_north = np.append(self.std_dev_pos_north, other.std_dev_pos_north)
        self.time_gps = np.append(self.time_gps, other.time_gps)
        self.traveled = np.append(self.traveled, other.traveled)
        self.vel_down = np.append(self.vel_down, other.vel_down)
        self.vel_forward = np.append(self.vel_forward, other.vel_forward)
        self.vel_lateral = np.append(self.vel_lateral, other.vel_lateral)
        return self

    def get_idx(self, start_idx, end_idx=None) -> "OXTSData":
        if end_idx is None:
            end_idx = start_idx + 1

        if type(start_idx) == int:
            return OXTSData(
                acceleration_x=self.acceleration_x[start_idx:end_idx],
                acceleration_y=self.acceleration_y[start_idx:end_idx],
                acceleration_z=self.acceleration_z[start_idx:end_idx],
                angular_rate_x=self.angular_rate_x[start_idx:end_idx],
                angular_rate_y=self.angular_rate_y[start_idx:end_idx],
                angular_rate_z=self.angular_rate_z[start_idx:end_idx],
                ecef_x=self.ecef_x[start_idx:end_idx],
                ecef_y=self.ecef_y[start_idx:end_idx],
                ecef_z=self.ecef_z[start_idx:end_idx],
                heading=self.heading[start_idx:end_idx],
                leap_seconds=self.leap_seconds[start_idx:end_idx],
                pitch=self.pitch[start_idx:end_idx],
                pos_alt=self.pos_alt[start_idx:end_idx],
                pos_lat=self.pos_lat[start_idx:end_idx],
                pos_lon=self.pos_lon[start_idx:end_idx],
                roll=self.roll[start_idx:end_idx],
                std_dev_pos_east=self.std_dev_pos_east[start_idx:end_idx],
                std_dev_pos_north=self.std_dev_pos_north[start_idx:end_idx],
                time_gps=self.time_gps[start_idx:end_idx],
                traveled=self.traveled[start_idx:end_idx],
                vel_down=self.vel_down[start_idx:end_idx],
                vel_forward=self.vel_forward[start_idx:end_idx],
                vel_lateral=self.vel_lateral[start_idx:end_idx],
            )
        else:
            return OXTSData(
                acceleration_x=self.acceleration_x[start_idx],
                acceleration_y=self.acceleration_y[start_idx],
                acceleration_z=self.acceleration_z[start_idx],
                angular_rate_x=self.angular_rate_x[start_idx],
                angular_rate_y=self.angular_rate_y[start_idx],
                angular_rate_z=self.angular_rate_z[start_idx],
                ecef_x=self.ecef_x[start_idx],
                ecef_y=self.ecef_y[start_idx],
                ecef_z=self.ecef_z[start_idx],
                heading=self.heading[start_idx],
                leap_seconds=self.leap_seconds[start_idx],
                pitch=self.pitch[start_idx],
                pos_alt=self.pos_alt[start_idx],
                pos_lat=self.pos_lat[start_idx],
                pos_lon=self.pos_lon[start_idx],
                roll=self.roll[start_idx],
                std_dev_pos_east=self.std_dev_pos_east[start_idx],
                std_dev_pos_north=self.std_dev_pos_north[start_idx],
                time_gps=self.time_gps[start_idx],
                traveled=self.traveled[start_idx],
                vel_down=self.vel_down[start_idx],
                vel_forward=self.vel_forward[start_idx],
                vel_lateral=self.vel_lateral[start_idx],
            )

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


def transform_points(points: np.ndarray, pose: Pose) -> np.ndarray:
    """Transform points from the ego vehicle frame to the global frame.

    Args:
        points: points in the ego vehicle frame to transform, shape: (N, 3)
        pose: ego vehicle pose at the timestamp of the points

    Returns:
        points in the global frame, shape: (N, 3)
    """
    return points @ pose.rotation_matrix.T + pose.translation


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

    def transform(self, pose: Pose) -> "LidarData":
        from zod.utils.geometry import _transform_points

        """Transform the lidar data to a new pose.

        Args:
            pose: The new pose to transform the lidar data to.

        Returns:
            The transformed lidar data.
        """
        self.points = _transform_points(self.points, pose.transform)
        return self

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

"""ZOD dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, List, Union

import numpy as np
from PIL import Image

from zod.anno.parser import ANNOTATION_PARSERS
from zod.constants import AnnotationProject
from zod.utils.utils import parse_datetime_from_filename

from ._serializable import JSONSerializable
from .geometry import Pose

# we want to remove the points that get returns from the ego vehicle.
# we do this by removing all points that are within the box defined by
# EGO_RETURNS_BOX. This box is defined in the LiDAR frame.
EGO_RETURNS_BOX = np.array([[-1.5, -3.0, -1.5], [1.5, 3.0, 0.5]])


@dataclass
class LidarData:
    """A class describing the lidar data."""

    points: np.ndarray  # (N, 3) float32
    timestamps: np.ndarray  # (N,) float64 epoch timestamp pointwise
    intensity: np.ndarray  # (N,) uint8
    diode_idx: np.ndarray  # (N,) uint8
    core_timestamp: float  # core epoch timestamp

    def copy(self) -> LidarData:
        """Return a copy of the lidar data."""
        return LidarData(
            points=self.points.copy(),
            timestamps=self.timestamps.copy(),
            intensity=self.intensity.copy(),
            diode_idx=self.diode_idx.copy(),
            core_timestamp=self.core_timestamp,
        )

    @classmethod
    def empty(cls) -> LidarData:
        """Create an empty lidar data object."""
        return cls(
            points=np.empty((0, 3), dtype=np.float32),
            timestamps=np.empty(0, dtype=np.float64),
            intensity=np.empty(0, dtype=np.uint8),
            diode_idx=np.empty(0, dtype=np.uint8),
            core_timestamp=0.0,
        )

    @classmethod
    def from_npy(cls, path: str, remove_ego_lidar_returns: bool = True) -> LidarData:
        """Load lidar data from a .npy file."""
        data = np.load(path)
        core_timestamp = parse_datetime_from_filename(path).timestamp()
        if remove_ego_lidar_returns:
            ego_returns_mask = np.ones(len(data), dtype=np.bool_)
            for idim, dim in enumerate(("x", "y", "z")):
                ego_returns_mask &= np.logical_and(
                    data[dim] > EGO_RETURNS_BOX[0, idim], data[dim] < EGO_RETURNS_BOX[1, idim]
                )
            data = data[~ego_returns_mask]
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
        self.points = (self.points[..., None, :] @ rotations.swapaxes(-2, -1) + translations[..., None, :]).squeeze(-2)

    def extend(self, *other: LidarData):
        """Extend this LidarData with data from another LidarData object.

        Args:
            other: The other LidarData object.
        """
        self.points = np.vstack((self.points, *(o.points for o in other)))
        self.timestamps = np.hstack((self.timestamps, *(o.timestamps for o in other)))
        self.intensity = np.hstack((self.intensity, *(o.intensity for o in other)))
        self.diode_idx = np.hstack((self.diode_idx, *(o.diode_idx for o in other)))
        # Core timestamp is the weighted average
        self.core_timestamp = (
            self.core_timestamp * len(self.timestamps) + sum(o.core_timestamp * len(o.timestamps) for o in other)
        ) / (len(self.timestamps) + sum(len(o.timestamps) for o in other))

    def __eq__(self, other: LidarData) -> Union[bool, np.bool_]:
        """Check if two LidarData objects are equal."""
        return (
            np.allclose(self.points, other.points)
            and np.allclose(self.timestamps, other.timestamps)
            and np.allclose(self.intensity, other.intensity)
            and np.allclose(self.diode_idx, other.diode_idx)
            and np.isclose(self.core_timestamp, other.core_timestamp)
        )


@dataclass
class RadarData:
    """A class describing the radar data."""

    radar_range: np.ndarray  # (N,) float32
    azimuth_angle: np.ndarray  # (N,) float32
    elevation_angle: np.ndarray  # (N,) float32
    range_rate: np.ndarray  # (N,) float32
    amplitude: np.ndarray  # (N,) float32
    validity: np.ndarray  # (N,) int8
    mode: np.ndarray  # (N,) uint8
    quality: np.ndarray  # (N,) uint8
    scan_index: np.ndarray  # (N,) uint32
    timestamp: np.ndarray  # (N,) int64

    def copy(self) -> RadarData:
        """Return a copy of the radar data."""
        return RadarData(
            radar_range=self.radar_range.copy(),
            azimuth_angle=self.azimuth_angle.copy(),
            elevation_angle=self.elevation_angle.copy(),
            range_rate=self.range_rate.copy(),
            amplitude=self.amplitude.copy(),
            validity=self.validity.copy(),
            mode=self.mode.copy(),
            quality=self.quality.copy(),
            scan_index=self.scan_index.copy(),
            timestamp=self.timestamp,
        )

    @classmethod
    def empty(cls) -> RadarData:
        """Create an empty radar data object."""
        return cls(
            radar_range=np.empty(0, dtype=np.float32),
            azimuth_angle=np.empty(0, dtype=np.float32),
            elevation_angle=np.empty(0, dtype=np.float32),
            range_rate=np.empty(0, dtype=np.float32),
            amplitude=np.empty(0, dtype=np.float32),
            validity=np.empty(0, dtype=np.int8),
            mode=np.empty(0, dtype=np.uint8),
            quality=np.empty(0, dtype=np.uint8),
            scan_index=np.empty(0, dtype=np.uint32),
            timestamp=0,
        )

    @classmethod
    def from_npy(cls, path: str) -> RadarData:
        """Load radar data from a .npy file.

        Args:
            path: Path to the file we are loading the data from."""
        data = np.load(path)
        return cls(
            radar_range=data["radar_range"],
            azimuth_angle=data["azimuth_angle"],
            elevation_angle=data["elevation_angle"],
            range_rate=data["range_rate"],
            amplitude=data["amplitude"],
            validity=data["validity"],
            mode=data["mode"],
            quality=data["quality"],
            scan_index=data["scan_index"],
            timestamp=data["timestamp"],
        )

    def to_npy(self, path: str) -> None:
        """Save radar data to a .npy file in the same format as is used for loading.

        Args:
            path: Path of the file we are saving the data in."""
        data = np.empty(
            len(self.radar_range),
            dtype=[
                ("scan_index", np.uint32),
                ("timestamp", np.int64),
                ("radar_range", np.float32),
                ("azimuth_angle", np.float32),
                ("elevation_angle", np.float32),
                ("range_rate", np.float32),
                ("amplitude", np.float32),
                ("validity", np.int8),
                ("mode", np.uint8),
                ("quality", np.uint8),
            ],
        )

        data["radar_range"] = self.radar_range
        data["azimuth_angle"] = self.azimuth_angle
        data["elevation_angle"] = self.elevation_angle
        data["range_rate"] = self.range_rate
        data["amplitude"] = self.amplitude
        data["validity"] = self.validity
        data["mode"] = self.mode
        data["quality"] = self.quality
        data["scan_index"] = self.scan_index
        if len(self.timestamp) == 1:
            data["timestamp"] = self.timestamp
        else:
            times = np.empty(len(self.radar_range), dtype=np.int64)
            for i in range(len(self.timestamp)):
                times[self.scan_index == i] = self.timestamp[i]
            data["timestamp"] = times

        np.save(path, data)

    def get_cartesian_coordinates(self) -> np.ndarray:
        """Convert radar data to cartesian coordinates with shape (N x 3)."""
        x = self.radar_range * np.cos(self.elevation_angle) * np.cos(self.azimuth_angle)
        y = self.radar_range * np.cos(self.elevation_angle) * np.sin(self.azimuth_angle)
        z = self.radar_range * np.sin(self.elevation_angle)
        return np.vstack((x, y, z)).T

    def extend(self, *other: RadarData):
        """Extend this RadarData with data from another RadarData object.

        Args:
            other: The other RadarData object.
        """
        self.radar_range = np.hstack((self.radar_range, *(o.radar_range for o in other)))
        self.azimuth_angle = np.hstack((self.azimuth_angle, *(o.azimuth_angle for o in other)))
        self.elevation_angle = np.hstack((self.elevation_angle, *(o.elevation_angle for o in other)))
        self.range_rate = np.hstack((self.range_rate, *(o.range_rate for o in other)))
        self.amplitude = np.hstack((self.amplitude, *(o.amplitude for o in other)))
        self.validity = np.hstack((self.validity, *(o.validity for o in other)))
        self.mode = np.hstack((self.mode, *(o.mode for o in other)))
        self.quality = np.hstack((self.quality, *(o.quality for o in other)))
        self.scan_index = np.hstack((self.scan_index, *(o.scan_index for o in other)))
        self.timestamp = np.vstack((self.timestamp, *(o.timestamp for o in other)))

    def __eq__(self, other: RadarData) -> Union[bool, np.bool_]:
        """Check if two RadarData objects are equal.

        Args:
            other: The other RadarData object."""
        return (
            np.allclose(self.radar_range, other.radar_range)
            and np.allclose(self.azimuth_angle, other.azimuth_angle)
            and np.allclose(self.elevation_angle, other.elevation_angle)
            and np.allclose(self.range_rate, other.range_rate)
            and np.allclose(self.amplitude, other.amplitude)
            and np.allclose(self.validity, other.validity)
            and np.allclose(self.mode, other.mode)
            and np.allclose(self.quality, other.quality)
            and np.allclose(self.scan_index, other.scan_index)
            and np.isclose(self.timestamp, other.timestamp)
        )


@dataclass
class SensorFrame(JSONSerializable):
    """Class to store sensor information."""

    filepath: str
    time: datetime

    def read(self) -> Any:
        """Read the data from the file."""
        raise NotImplementedError


@dataclass
class LidarFrame(SensorFrame):
    """Class to store information about a lidar frame."""

    is_compensated: bool  # Whether the cloud is pointwise compensated

    def read(self, remove_ego_lidar_returns: bool = True) -> LidarData:
        """Read the point cloud."""
        return LidarData.from_npy(self.filepath, remove_ego_lidar_returns=remove_ego_lidar_returns)


@dataclass
class RadarFrames(JSONSerializable):
    """Class to store information about a radar sequence file."""

    filepath: str
    time: datetime  # time of the sequence key frame

    def read(self) -> RadarData:
        """Read the radar data."""
        return RadarData.from_npy(self.filepath)


@dataclass
class CameraFrame(SensorFrame):
    """Class to store information about a camera frame."""

    height: int = 2168
    width: int = 3848

    def read(self) -> np.ndarray:
        """Read the image."""
        return np.array(Image.open(self.filepath))

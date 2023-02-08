"""ZOD dataclasses."""
from dataclasses import dataclass
from datetime import datetime
from typing import Any, List, Union

import numpy as np
from PIL import Image

from zod.constants import AnnotationProject
from zod.utils.annotation_parser import ANNOTATION_PARSERS
from zod.utils.utils import parse_datetime_from_filename

from ._serializable import JSONSerializable
from .geometry import Pose


@dataclass
class LidarData:
    """A class describing the lidar data."""

    points: np.ndarray  # (N, 3) float32
    timestamps: np.ndarray  # (N,) float64 epoch timestamp pointwise
    intensity: np.ndarray  # (N,) uint8
    diode_idx: np.ndarray  # (N,) uint8
    core_timestamp: float  # core epoch timestamp

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
            core_timestamp=0.0,
        )

    @classmethod
    def from_npy(cls, path: str) -> "LidarData":
        """Load lidar data from a .npy file."""
        data = np.load(path)
        core_timestamp = parse_datetime_from_filename(path).timestamp()
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
        ).squeeze(-2)

    def extend(self, *other: "LidarData"):
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
            self.core_timestamp * len(self.timestamps)
            + sum(o.core_timestamp * len(o.timestamps) for o in other)
        ) / (len(self.timestamps) + sum(len(o.timestamps) for o in other))

    def __eq__(self, other: "LidarData") -> bool:
        """Check if two LidarData objects are equal."""
        return (
            np.allclose(self.points, other.points)
            and np.allclose(self.timestamps, other.timestamps)
            and np.allclose(self.intensity, other.intensity)
            and np.allclose(self.diode_idx, other.diode_idx)
            and np.isclose(self.core_timestamp, other.core_timestamp)
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

    def read(self) -> LidarData:
        """Read the point cloud."""
        return LidarData.from_npy(self.filepath)


@dataclass
class CameraFrame(SensorFrame):
    """Class to store information about a camera frame."""

    height: int = 2168
    width: int = 3848

    def read(self) -> np.ndarray:
        """Read the image."""
        return np.array(Image.open(self.filepath))


@dataclass
class AnnotationFrame(SensorFrame):
    """Class to store information about an annotation frame."""

    project: AnnotationProject

    def read(self) -> List[Any]:
        """Read (and parse) the annotation json."""
        return ANNOTATION_PARSERS[self.project](self.filepath)

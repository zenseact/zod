from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Tuple, Union

import h5py
import numpy as np
import quaternion


OXTS_TIMESTAMP_OFFSET = datetime(1980, 1, 6, tzinfo=timezone.utc).timestamp()


@dataclass
class EgoMotion:
    pose: np.ndarray  # Nx4x4
    velocity: np.ndarray  # Nx3
    acceleration: np.ndarray  # Nx3
    angular_rate: np.ndarray  # Nx3
    timestamp: np.ndarray  # N - epoch time in seconds (float64)
    origin_lat_lon: Tuple[float, float]  # (lat, lon) of the origin (which poses are relative to)

    def __len__(self) -> int:
        return len(self.timestamp)

    def get_poses(self, target_ts: Union[np.ndarray, float, np.float64]) -> np.ndarray:
        """Interpolate neighbouring poses to find pose for each target timestamp.
        Args:
            target_ts: [N] timestamps for which to find a pose by interpolation.
        Return:
            [N, 4, 4] array of interpolated poses for each timestamp.
        """
        selfmin, selfmax = np.min(self.timestamp), np.max(self.timestamp)
        assert (selfmin <= np.min(target_ts)) and (
            selfmax > np.max(target_ts)
        ), f"targets not between pose timestamps, must be [{selfmin}, {selfmax})"
        closest_idxs = self.timestamp.searchsorted(
            target_ts, side="right", sorter=self.timestamp.argsort()
        )
        time_diffs = target_ts - self.timestamp[closest_idxs - 1]
        total_times = self.timestamp[closest_idxs] - self.timestamp[closest_idxs - 1]
        fractions = time_diffs / total_times
        return interpolate_transforms(
            self.pose[closest_idxs - 1], self.pose[closest_idxs], fractions
        )

    @classmethod
    def from_sequence_oxts(cls, file: h5py.Group) -> "EgoMotion":
        return cls(
            pose=file["poses"][()],
            acceleration=np.stack(
                [file["accelerationX"], file["accelerationY"], file["accelerationZ"]], axis=1
            ),
            # TODO: figure out what order should be here,
            velocity=np.stack([file["velDown"], file["velForward"], file["velLateral"]], axis=1),
            angular_rate=np.stack(
                [file["angularRateX"], file["angularRateY"], file["angularRateZ"]], axis=1
            ),
            timestamp=OXTS_TIMESTAMP_OFFSET + file["timestamp"][()] + file["leapSeconds"][()],
            origin_lat_lon=(file["posLat"][0], file["posLon"][0]),
        )

    @classmethod
    def from_frame_oxts(cls, file: h5py.Group) -> "EgoMotion":
        raise NotImplementedError("William fix hehe")

    @classmethod
    def from_json(cls, json: dict) -> "EgoMotion":
        raise NotImplementedError("generate these first")

    def to_json(self) -> dict:
        raise NotImplementedError("generate these first")


def interpolate_transforms(transform_1: np.ndarray, transform_2: np.ndarray, fractions: np.ndarray):
    """Interpolate between two transforms.

    Args:
        transform_1: First transform (Mx4x4).
        transform_2: Second transform (Mx4x4).
        fraction: Fraction of interpolation (M).

    Returns:
        Interpolated transform(s) (Mx4x4).
    """
    R_a, t_a = transform_1[..., :3, :3], transform_1[..., :3, 3]
    R_b, t_b = transform_2[..., :3, :3], transform_2[..., :3, 3]
    # Alternative would be to convert to euler angles and interpolate, but this is more accurate.
    R_interpolated = quaternion.as_rotation_matrix(
        np.slerp_vectorized(
            quaternion.from_rotation_matrix(R_a),
            quaternion.from_rotation_matrix(R_b),
            fractions,
        )
    )
    t_interp = t_a + fractions[..., None] * (t_b - t_a)
    transform = np.concatenate(
        [
            np.concatenate([R_interpolated, t_interp[..., None]], axis=-1),
            np.broadcast_to(np.array([0, 0, 0, 1]), R_interpolated.shape[:-2] + (1, 4)),
        ],
        axis=-2,
    )
    return transform

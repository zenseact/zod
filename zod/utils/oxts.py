from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Tuple, Union

import h5py
import numpy as np
import quaternion


OXTS_TIMESTAMP_OFFSET = datetime(1980, 1, 6, tzinfo=timezone.utc).timestamp()


@dataclass
class EgoMotion:
    poses: np.ndarray  # Nx4x4
    velocities: np.ndarray  # Nx3
    accelerations: np.ndarray  # Nx3
    angular_rates: np.ndarray  # Nx3
    timestamps: np.ndarray  # N - epoch time in seconds (float64)
    origin_lat_lon: Tuple[float, float]  # (lat, lon) of the origin (which pose are relative to)

    def __len__(self) -> int:
        return len(self.timestamps)

    def get_poses(self, target_ts: Union[np.ndarray, float, np.float64]) -> np.ndarray:
        """Interpolate neighbouring poses to find pose for each target timestamp.
        Args:
            target_ts: [N] timestamps for which to find a pose by interpolation.
        Return:
            [N, 4, 4] array of interpolated poses for each timestamp.
        """
        selfmin, selfmax = np.min(self.timestamps), np.max(self.timestamps)
        assert (selfmin <= np.min(target_ts)) and (
            selfmax > np.max(target_ts)
        ), f"targets not between pose timestamps, must be [{selfmin}, {selfmax})"
        # TODO: check if the timestamp exists, then we don't need to interpolate
        # else interpolate according to below.
        closest_idxs = self.timestamps.searchsorted(
            target_ts, side="right", sorter=self.timestamps.argsort()
        )
        time_diffs = target_ts - self.timestamps[closest_idxs - 1]
        total_times = self.timestamps[closest_idxs] - self.timestamps[closest_idxs - 1]
        fractions = time_diffs / total_times
        return interpolate_transforms(
            self.poses[closest_idxs - 1], self.poses[closest_idxs], fractions
        )

    def interpolate(self, timestamps: np.ndarray) -> "EgoMotion":
        """Interpolate ego motion to find ego motion for each target timestamp."""
        poses = self.get_poses(timestamps)
        velocities = interpolate_vectors(self.velocities, self.timestamps, timestamps)
        accelerations = interpolate_vectors(self.accelerations, self.timestamps, timestamps)
        angular_rates = interpolate_vectors(self.angular_rates, self.timestamps, timestamps)

        return EgoMotion(
            poses=poses,
            velocities=velocities,
            accelerations=accelerations,
            angular_rates=angular_rates,
            timestamps=timestamps,
            origin_lat_lon=self.origin_lat_lon,
        )

    @classmethod
    def from_sequence_oxts(cls, oxts_path: str) -> "EgoMotion":
        with h5py.File(oxts_path, "r") as file:
            return cls(
                poses=file["poses"][()],
                accelerations=np.stack(
                    [file["accelerationX"], file["accelerationY"], file["accelerationZ"]], axis=1
                ),
                # TODO: figure out what order should be here,
                velocities=np.stack(
                    [file["velDown"], file["velForward"], file["velLateral"]], axis=1
                ),
                angular_rates=np.stack(
                    [file["angularRateX"], file["angularRateY"], file["angularRateZ"]], axis=1
                ),
                timestamps=OXTS_TIMESTAMP_OFFSET + file["timestamp"][()] + file["leapSeconds"][()],
                origin_lat_lon=(file["posLat"][0], file["posLon"][0]),
            )

    @classmethod
    def from_frame_oxts(cls, file: h5py.Group) -> "EgoMotion":
        return cls(
            poses=get_poses_from_oxts(file),
            accelerations=np.stack(
                [file["accelerationX"], file["accelerationY"], file["accelerationZ"]], axis=1
            ),
            # TODO: figure out what order should be here,
            velocities=np.stack([file["velDown"], file["velForward"], file["velLateral"]], axis=1),
            angular_rates=np.stack(
                [file["angularRateX"], file["angularRateY"], file["angularRateZ"]], axis=1
            ),
            timestamps=OXTS_TIMESTAMP_OFFSET + file["timestamp"][()] + file["leapSeconds"][()],
            origin_lat_lon=(file["posLat"][0], file["posLon"][0]),
        )

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


def interpolate_vectors(
    values: np.ndarray, timestamps: np.ndarray, target_timestamps: np.ndarray
) -> np.ndarray:
    """Interpolate vectors to find vector"""
    assert np.all(
        timestamps >= np.min(target_timestamps)
    ), "target timestamps must be within timestamps"
    assert np.all(
        timestamps <= np.max(target_timestamps)
    ), "target timestamps must be within timestamps"
    return np.interp(target_timestamps, timestamps, values, left=values[0], right=values[-1])


def get_poses_from_oxts(file: h5py.Group) -> np.ndarray:
    """Get poses from oxts file.

    Args:
        file: oxts file.

    Returns:
        [N, 4, 4] array of poses.
    """
    raise NotImplementedError

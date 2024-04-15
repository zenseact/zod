from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Tuple, Union

import h5py
import numpy as np
import quaternion
from scipy.interpolate import interp1d

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

    def get_poses(self, target_ts: Union[np.ndarray, float]) -> np.ndarray:
        """Interpolate neighbouring poses to find pose for each target timestamp.
        Args:
            target_ts: [N] timestamps for which to find a pose by interpolation.
        Return:
            [N, 4, 4] array of interpolated poses for each timestamp.
        """
        selfmin, selfmax = np.min(self.timestamps), np.max(self.timestamps)
        assert (selfmin <= np.min(target_ts)) and (
            selfmax >= np.max(target_ts)
        ), f"targets not between pose timestamps, must be [{selfmin}, {selfmax}]"

        if np.isin(target_ts, self.timestamps).all():
            return self.poses[self.timestamps.searchsorted(target_ts)]

        closest_idxs = self.timestamps.searchsorted(target_ts, side="right", sorter=self.timestamps.argsort())

        # if the target timestamp is exactly the same as the largest timestamp
        # then the searchsorted will return the length of the array, which is
        # out of bounds. The assert above ensures that the target timestamp is
        # less or equal than the largest timestamp, so we can just clip the index
        # to the last element.
        closest_idxs = np.clip(closest_idxs, 1, len(self.timestamps) - 1)

        time_diffs = target_ts - self.timestamps[closest_idxs - 1]
        total_times = self.timestamps[closest_idxs] - self.timestamps[closest_idxs - 1]
        fractions = time_diffs / total_times
        return interpolate_transforms(self.poses[closest_idxs - 1], self.poses[closest_idxs], fractions)

    def interpolate(self, timestamps: np.ndarray) -> EgoMotion:
        """Interpolate ego motion to find ego motion for each target timestamp.

        Args:
            timestamps: [N] timestamps for which to find ego motion by interpolation.
        Return:
            EgoMotion object with interpolated ego motion for each timestamp.
        """
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
    def from_oxts_path(cls, file_path: str) -> EgoMotion:
        """Load ego motion from a sequence or frame oxts file."""
        with h5py.File(file_path, "r") as file:
            return cls(
                poses=file["poses"][()],
                accelerations=np.stack([file["accelerationX"], file["accelerationY"], file["accelerationZ"]], axis=1),
                velocities=np.stack([file["velForward"][()], file["velLateral"][()], -file["velDown"][()]], axis=1),
                angular_rates=np.stack([file["angularRateX"], file["angularRateY"], file["angularRateZ"]], axis=1),
                timestamps=OXTS_TIMESTAMP_OFFSET + file["timestamp"][()] + file["leapSeconds"][()],
                origin_lat_lon=(file["posLat"][0], file["posLon"][0]),
            )

    @classmethod
    def from_json_path(cls, json_path: str) -> EgoMotion:
        """Load ego motion from a json file."""
        with open(json_path, "r") as file:
            data = json.load(file)
        return cls(
            poses=np.array(data["poses"]),
            velocities=np.array(data["velocities"]),
            accelerations=np.array(data["accelerations"]),
            angular_rates=np.array(data["angular_rates"]),
            timestamps=np.array(data["timestamps"]),
            origin_lat_lon=tuple(data["origin_lat_lon"]),
        )

    def to_json(self) -> dict:
        """Save ego motion to a json file."""
        return {
            "poses": self.poses.tolist(),
            "velocities": self.velocities.tolist(),
            "accelerations": self.accelerations.tolist(),
            "angular_rates": self.angular_rates.tolist(),
            "timestamps": self.timestamps.tolist(),
            "origin_lat_lon": self.origin_lat_lon,
        }


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


def interpolate_vectors(values: np.ndarray, source_timestamps: np.ndarray, target_timestamps: np.ndarray) -> np.ndarray:
    """Interpolate vectors to find vector for each (sorted) target timestamp.

    Args:
        values: [N, X] array of values.
        source_timestamps: [N] array of timestamps.
        target_timestamps: [M] array of timestamps.
    Returns:
        [M, X] array of interpolated values.
    """
    source_min_ts, source_max_ts = np.min(source_timestamps), np.max(source_timestamps)
    target_min_ts, target_max_ts = np.min(target_timestamps), np.max(target_timestamps)
    assert source_min_ts <= target_min_ts, "Target timestamps must be after source timestamps"
    assert source_max_ts >= target_max_ts, "Target timestamps must be before source timestamps"
    # Interpolate values to find values for each target timestamp without scipy
    return interp1d(source_timestamps, values, axis=0)(target_timestamps)

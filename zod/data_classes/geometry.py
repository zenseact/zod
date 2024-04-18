from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyquaternion import Quaternion


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
    def inverse(self) -> Pose:
        """Return the inverse of the pose."""
        return Pose(np.linalg.inv(self.transform))

    @classmethod
    def from_translation_rotation(cls, translation: np.ndarray, rotation_matrix: np.ndarray) -> Pose:
        """Create a pose from a translation and a rotation."""
        transform = np.eye(4, 4)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = translation
        return cls(transform)

    @classmethod
    def identity(cls) -> Pose:
        """Create an identity pose."""
        return cls(np.eye(4, 4))

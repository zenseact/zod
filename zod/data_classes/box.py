"""ZOD Object detection containers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from pyquaternion import Quaternion

from zod.constants import EGO, Camera, CoordinateFrame
from zod.data_classes import Calibration, Pose
from zod.utils.geometry import project_3d_to_2d_kannala, unproject_2d_to_3d_kannala


@dataclass
class Box3D:
    """Class to store a 3D bounding box.

    The box is defined by its center, size, orientation in the given coordinate frame.
    The size is defined as (length, width, height).
    """

    center: np.ndarray  # x, y, z
    size: np.ndarray  # L, W, H
    orientation: Quaternion
    frame: CoordinateFrame

    def _transform(self, transform: Optional[Pose], new_frame: CoordinateFrame):
        if transform is not None:
            self.center = transform.rotation_matrix @ self.center
            self.center += transform.translation
            self.orientation = transform.rotation * self.orientation
            self.frame = new_frame

    def _transform_inv(self, transform: Optional[Pose], new_frame: CoordinateFrame):
        if transform is not None:
            self.center -= transform.translation
            self.center = transform.rotation_matrix.T @ self.center
            self.orientation = transform.rotation.inverse * self.orientation
            self.frame = new_frame

    def convert_to(self, frame: CoordinateFrame, calib: Calibration):
        if frame == self.frame:
            return
        self._transform(calib.get_extrinsics(self.frame), new_frame=EGO)
        self._transform_inv(calib.get_extrinsics(frame), new_frame=frame)

    def copy(self):
        return Box3D(
            center=self.center.copy(),
            size=self.size.copy(),
            orientation=Quaternion(self.orientation),
            frame=self.frame,
        )

    @property
    def corners(self) -> np.ndarray:
        """Get the corners of the bounding box in the current frame.

        Order of points are:
         - rear left bottom
         - rear right bottom
         - front right bottom
         - front left bottom
         - rear left top
         - rear right top
         - front right top
         - front left top
        """
        # Get the 3d corners of the box
        corners = np.array(
            [
                [-0.5, -0.5, -0.5],
                [0.5, -0.5, -0.5],
                [0.5, 0.5, -0.5],
                [-0.5, 0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [0.5, -0.5, 0.5],
                [0.5, 0.5, 0.5],
                [-0.5, 0.5, 0.5],
            ]
        )
        corners *= self.size
        corners = self.orientation.rotation_matrix @ corners.T
        corners += self.center.reshape((-1, 1))
        return corners.T

    @property
    def corners_bev(self) -> np.ndarray:
        """Get the corners of the 3D bounding box where the z-dimension is (birds-eye-view)."""
        return self.corners[:4, :2]

    def project_into_camera(self, calib: Calibration) -> np.ndarray:
        """Project 3D bounding box into 2D image plane.

        Return an array (9, 2) the box center and the box corners in the imageplane
        of the given camera. The order of the points is:
            - center point
            - rear left bottom
            - rear right bottom
            - front right bottom
            - front left bottom
            - rear left top
            - rear right top
            - front right top
            - front left top
        """
        if not isinstance(self.frame, Camera):
            raise ValueError(f"Cannot project box into frame {self.frame}")

        # Concatenate the corners and center of the box
        points = np.concatenate([self.center.reshape((1, -1)), self.corners], axis=0)

        # Project center into camera
        pos2d = project_3d_to_2d_kannala(
            points,
            calib.cameras[self.frame].intrinsics,
            calib.cameras[self.frame].distortion,
        )
        return pos2d

    def __eq__(self, __o: Box3D) -> bool:
        b1 = np.allclose(self.center, __o.center)
        b2 = np.allclose(self.size, __o.size)
        b3 = self.orientation == __o.orientation
        b4 = self.frame == __o.frame
        return b1 and b2 and b3 and b4


@dataclass
class Box2D:
    """2D Bounding Box container."""

    xyxy: np.ndarray  # xmin, ymin, xmax, ymax
    frame: Camera

    @classmethod
    def from_points(cls, points: List[List[float]], frame=Camera.FRONT) -> Box2D:
        """Compute outer points from a polygon.

        Args:
            points (list): list of points shaping bounding box and the properties.

        """
        xmin = ymin = 999999
        xmax = ymax = -1
        for point_x, point_y in points:
            xmin = min(xmin, point_x)
            xmax = max(xmax, point_x)
            ymin = min(ymin, point_y)
            ymax = max(ymax, point_y)
        return cls(xyxy=np.array([xmin, ymin, xmax, ymax], dtype=np.float32), frame=frame)

    @property
    def xywh(self) -> np.ndarray:
        """Get the box in xywh format."""
        return self.xyxy - np.array([0, 0, self.xyxy[0], self.xyxy[1]], dtype=self.xyxy.dtype)

    @property
    def area(self) -> float:
        """Bounding Box 2D area.

        Returns:
            Area of the Bounding Box.
        """
        return float(np.prod(self.xyxy[2:] - self.xyxy[:2]))

    @property
    def center(self) -> np.ndarray:
        """Bounding Box 2D center.

        Returns:
            Center of the Bounding Box.
        """
        return (self.xyxy[:2] + self.xyxy[2:]) / 2

    @property
    def dimension(self) -> np.ndarray:
        """Bounding Box 2D dimension.

        Returns:
            Dimension of the Bounding Box.
        """
        return self.xyxy[2:] - self.xyxy[:2]

    @property
    def corners(
        self,
    ) -> np.ndarray:
        """Bounding Box 2D corners.

        Returns:
            array (4, 2) of the corners of the Bounding Box.
            - top left
            - top right
            - bottom right
            - bottom left
        """
        return np.stack(
            [
                self.xyxy[[0, 1]],
                self.xyxy[[2, 1]],
                self.xyxy[[2, 3]],
                self.xyxy[[0, 3]],
            ]
        )

    @property
    def xmin(self) -> int:
        """Get the (rounded) left pixel coordinate."""
        return int(self.xyxy[0].round())

    @property
    def ymin(self) -> int:
        """Get the (rounded) top pixel coordinate."""
        return int(self.xyxy[1].round())

    @property
    def xmax(self) -> int:
        """Get the (rounded) right pixel coordinate."""
        return int(self.xyxy[2].round())

    @property
    def ymax(self) -> int:
        """Get the (rounded) bottom pixel coordinate."""
        return int(self.xyxy[3].round())

    def crop_from_image(
        self,
        image: np.ndarray,
        padding: Optional[Tuple[int, int]],
        padding_factor: Optional[float],
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Crop out the Bounding Box from an image.

        Args:
            image: Image to crop out the Bounding Box from.
            padding: Tuple of padding in x and y direction.
            padding_factor: Factor to multiply the padding with.

        Returns:
            Cropped image.
            tuple: padding in all four directions | (left,  top,  right, bottom)
        """

        assert not (padding is not None and padding_factor is not None), (
            "Cannot specify both padding and padding_factor"
        )

        if padding is not None:
            padding_x, padding_y = padding
        elif padding_factor is not None:
            padding_x = int(self.dimension[0] * padding_factor)
            padding_y = int(self.dimension[1] * padding_factor)
        else:
            padding_x = 0
            padding_y = 0

        xmin = max(0, self.xmin - padding_x)
        ymin = max(0, self.ymin - padding_y)
        xmax = min(image.shape[1], self.xmax + padding_x)
        ymax = min(image.shape[0], self.ymax + padding_y)

        return image[ymin:ymax, xmin:xmax], (
            self.xmin - xmin,
            self.ymin - ymin,
            xmax - self.xmax,
            ymax - self.ymax,
        )

    def points_in_box(self, points: np.ndarray) -> np.ndarray:
        """Check if a point is inside the Bounding Box.

        Args:
            point: points to check. (n, 2)

        Returns:
            True if point is inside the Bounding Box.
        """

        return points >= self.xyxy[:2] & points <= self.xyxy[2:]

    def get_3d_frustum(
        self,
        calibration: Calibration,
        frame: Optional[CoordinateFrame] = None,
        min_depth: float = 0.0,
        max_depth: float = 500.0,
    ) -> np.ndarray:
        """Get the frustum of the bounding box in homogenous coordinates.

        The frustum is defined by the by having the 4 corners of the bounding box
        projected to the max_depth along with camera center (0, 0, 0).
        Args:
            calibration: Calibration data.
            frame: Frame to project the frustum into.
            max_depth: maximum depth of the frustum.

        Returns:
            frustum: (8, 4) array of the frustum. Note that is min_depth == 0.0,
            the first 4 points will correspond to camera center (0, 0, 0)
        """
        # Get the 2d bounding box corners
        corners = self.corners
        camera_calib = calibration.cameras[self.frame]
        if min_depth > 0:
            # Copy the corners
            corners = np.concatenate([corners, corners], axis=0)

            # Create the depth vector where the first four correspond to min_depth
            # and the last four correspond to max_depth
            depth = np.array([min_depth] * 4 + [max_depth] * 4)

        else:
            depth = np.array([max_depth] * 4)

        # Project the 2d corners to the max_depth using the calibration
        frustum = unproject_2d_to_3d_kannala(corners, camera_calib.intrinsics, camera_calib.undistortion, depth)

        if min_depth == 0.0:
            frustum = np.concatenate((np.zeros((4, 3)), frustum), axis=0)

        # Transform the frustum to the selected frame if needed
        if frame is not None:
            frustum = calibration.transform_points(
                points=frustum,
                from_frame=self.frame,
                to_frame=frame,
            )

        return frustum

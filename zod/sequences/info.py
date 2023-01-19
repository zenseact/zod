"""Sequence Information."""
import os.path as osp
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterator, List, Tuple

from zod.zod_dataclasses import JSONSerializable

from zod.zod_dataclasses.zod_dataclasses import CameraFrame, SensorFrame


@dataclass
class ZodSequenceInfo(JSONSerializable):
    """Class to store frame information."""

    sequence_id: str
    start_time: datetime
    end_time: datetime

    # These are chronologically ordered
    lidar_frames: Dict[str, List[SensorFrame]]
    camera_frames: Dict[str, List[CameraFrame]]
    oxts_path: str
    ego_motion_path: str
    calibration_path: str
    metadata_path: str

    def convert_paths_to_absolute(self, root_path: str):
        self.oxts_path = osp.join(root_path, self.oxts_path)
        self.calibration_path = osp.join(root_path, self.calibration_path)
        self.metadata_path = osp.join(root_path, self.metadata_path)
        self.ego_motion_path = osp.join(root_path, self.ego_motion_path)
        for lidar_frames in self.lidar_frames.values():
            for sensor_frame in lidar_frames:
                sensor_frame.filepath = osp.join(root_path, sensor_frame.filepath)
        for camera_frames in self.camera_frames.values():
            for sensor_frame in camera_frames:
                sensor_frame.filepath = osp.join(root_path, sensor_frame.filepath)

    def get_camera_lidar_map(
        self, camera: str, lidar: str
    ) -> Iterator[Tuple[CameraFrame, SensorFrame]]:
        """Iterate over all camera frames and their corresponding lidar frames.

        Args:
            camera: The camera to use. e.g., camera_front_blur
            lidar: The lidar to use. e.g., lidar_velodyne
        Yields:
            A tuple of the camera frame and the closest lidar frame.
        """
        assert (
            camera in self.camera_frames
        ), f"Camera {camera} not found. Available cameras: {self.camera_frames.keys()}"
        assert (
            lidar in self.lidar_frames
        ), f"Lidar {lidar} not found. Available lidars: {self.lidar_frames.keys()}"

        for camera_frame in self.camera_frames[camera]:
            # Get the closest lidar frame in time
            lidar_frame = min(
                self.lidar_frames[lidar],
                key=lambda lidar_frame: abs(lidar_frame.time - camera_frame.time),
            )
            yield camera_frame, lidar_frame

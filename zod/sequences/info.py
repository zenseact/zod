"""Sequence Information."""
import os.path as osp
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

from dataclass_wizard import JSONSerializable

from zod.utils.zod_dataclasses import CameraFrame, SensorFrame


@dataclass
class SequenceInformation(JSONSerializable):
    """Class to store frame information."""

    frame_id: str
    start_time: datetime
    end_time: datetime

    # these are chronologicaly ordered
    lidar_frames: Dict[str, List[SensorFrame]]
    camera_frames: Dict[str, List[CameraFrame]]
    oxts_path: str
    calibration_path: str
    metadata_path: str

    def convert_paths_to_absolute(self, root_path: str):
        self.oxts_path = osp.join(root_path, self.oxts_path)
        self.calibration_path = osp.join(root_path, self.calibration_path)
        self.metadata_path = osp.join(root_path, self.metadata_path)
        for lidar_frames in self.lidar_frames.values():
            for sensor_frame in lidar_frames:
                sensor_frame.filepath = osp.join(root_path, sensor_frame.filepath)
        for camera_frames in self.camera_frames.values():
            for sensor_frame in camera_frames:
                sensor_frame.filepath = osp.join(root_path, sensor_frame.filepath)

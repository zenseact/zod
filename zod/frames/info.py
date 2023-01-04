"""ZOD single-frames information."""
import os.path as osp
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Union

from dataclass_wizard import JSONSerializable

from zod.utils.zod_dataclasses import CameraFrame, SensorFrame


@dataclass
class FrameInformation(JSONSerializable):
    """Class to store frame information."""

    frame_id: str
    time: datetime

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
            osp.join(root_path, self.traffic_sign_annotation_path)
            if self.traffic_sign_annotation_path
            else None
        )
        self.ego_road_annotation_path = (
            osp.join(root_path, self.ego_road_annotation_path)
            if self.ego_road_annotation_path
            else None
        )
        self.object_detection_annotation_path = osp.join(
            root_path, self.object_detection_annotation_path
        )
        self.lane_markings_annotation_path = osp.join(root_path, self.lane_markings_annotation_path)
        self.road_condition_annotation_path = osp.join(
            root_path, self.road_condition_annotation_path
        )
        self.oxts_path = osp.join(root_path, self.oxts_path)
        self.calibration_path = osp.join(root_path, self.calibration_path)
        self.metadata_path = osp.join(root_path, self.metadata_path)
        for sensor_frame in self.lidar_frame.values():
            sensor_frame.filepath = osp.join(root_path, sensor_frame.filepath)
        for lidar_frames in self.previous_lidar_frames.values():
            for sensor_frame in lidar_frames:
                sensor_frame.filepath = osp.join(root_path, sensor_frame.filepath)
        for lidar_frames in self.future_lidar_frames.values():
            for sensor_frame in lidar_frames:
                sensor_frame.filepath = osp.join(root_path, sensor_frame.filepath)
        for sensor_frame in self.camera_frame.values():
            sensor_frame.filepath = osp.join(root_path, sensor_frame.filepath)

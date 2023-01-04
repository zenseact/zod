import os.path as osp
from dataclasses import dataclass
from datetime import datetime
from itertools import chain
from typing import Dict, Iterator, List, Tuple

from dataclass_wizard import JSONSerializable

from zod.dataclasses.zod_dataclasses import AnnotationFrame, CameraFrame, SensorFrame


@dataclass
class Information(JSONSerializable):
    """Base class for frame and sequence information."""

    id: str
    start_time: datetime
    end_time: datetime
    keyframe_time: datetime

    calibration_path: str
    ego_motion_path: str
    metadata_path: str
    oxts_path: str

    annotation_frames: Dict[str, List[AnnotationFrame]]
    camera_frames: Dict[str, List[CameraFrame]]
    lidar_frames: Dict[str, List[SensorFrame]]

    def convert_paths_to_absolute(self, root_path: str):
        self.calibration_path = osp.join(root_path, self.calibration_path)
        self.ego_motion_path = osp.join(root_path, self.ego_motion_path)
        self.metadata_path = osp.join(root_path, self.metadata_path)
        self.oxts_path = osp.join(root_path, self.oxts_path)

        for frame in chain(
            self.lidar_frames.values(), self.camera_frames.values(), self.annotation_frames.values()
        ):
            for sensor_frame in frame:
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
            # get the closest lidar frame in time
            lidar_frame = min(
                self.lidar_frames[lidar],
                key=lambda lidar_frame: abs(lidar_frame.time - camera_frame.time),
            )
            yield camera_frame, lidar_frame

    def get_keyframe_annotation(self, project: str) -> AnnotationFrame:
        raise NotImplementedError

    def get_keyframe_camera_frame(
        self, camera: str = "front", anonymization_mode: str = "blur"
    ) -> CameraFrame:
        raise NotImplementedError

    def get_keyframe_lidar_frame(self, lidar: str = "lidar_velodyne") -> SensorFrame:
        raise NotImplementedError

    def get_annotation(self, project: str, time: datetime) -> AnnotationFrame:
        raise NotImplementedError

    def get_camera_frame(
        self, time: datetime, camera: str = "front", anonymization_mode: str = "blur"
    ) -> CameraFrame:
        raise NotImplementedError

    def get_lidar_frame(self, time: datetime, lidar: str = "lidar_velodyne") -> SensorFrame:
        raise NotImplementedError

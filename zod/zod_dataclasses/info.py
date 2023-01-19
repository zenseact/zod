import os.path as osp
from dataclasses import dataclass
from datetime import datetime
from itertools import chain
from typing import Dict, Iterator, List, Optional, Tuple

from zod.zod_dataclasses import JSONSerializable

from zod.constants import AnnotationProject, Anonymization, Camera, Lidar
from zod.zod_dataclasses.zod_dataclasses import AnnotationFrame, CameraFrame, SensorFrame


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

    @property
    def all_frames(self) -> Iterator[SensorFrame]:
        """Iterate over all frames."""
        return chain(
            *self.lidar_frames.values(),
            *self.camera_frames.values(),
            *self.annotation_frames.values(),
        )

    def convert_paths_to_absolute(self, root_path: str):
        self.calibration_path = osp.join(root_path, self.calibration_path)
        self.ego_motion_path = osp.join(root_path, self.ego_motion_path)
        self.metadata_path = osp.join(root_path, self.metadata_path)
        self.oxts_path = osp.join(root_path, self.oxts_path)
        for frame in self.all_frames:
            frame.filepath = osp.join(root_path, frame.filepath)

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

    ### Keyframe accessors ###

    def get_keyframe_annotation(self, project: AnnotationProject) -> Optional[AnnotationFrame]:
        if project.value not in self.annotation_frames:
            return None
        if len(self.annotation_frames[project.value]) == 1:
            return self.annotation_frames[project.value][0]
        else:
            return self.get_annotation(self.keyframe_time, project)

    def get_keyframe_camera_frame(
        self,
        camera: Camera = Camera.front,
        anonymization: Anonymization = Anonymization.blur,
    ) -> CameraFrame:
        camera_name = f"{camera.value}_{anonymization.value}"
        if len(self.camera_frames[camera_name]) == 1:
            return self.camera_frames[camera_name][0]
        else:
            return min(
                self.camera_frames[camera_name],
                key=lambda camera_frame: abs(camera_frame.time - self.keyframe_time),
            )

    def get_keyframe_lidar_frame(self, lidar: Lidar = Lidar.velodyne) -> SensorFrame:
        return self.get_lidar_frame(self.keyframe_time, lidar)

    ### Timestamp accessors ###

    def get_annotation(
        self, time: datetime, project: AnnotationProject
    ) -> Optional[AnnotationFrame]:
        if project.value not in self.annotation_frames:
            return None
        return min(
            self.annotation_frames[project.value],
            key=lambda annotation_frame: abs(annotation_frame.time - time),
        )

    def get_camera_frame(
        self,
        time: datetime,
        camera: Camera = Camera.front,
        anonymization: Anonymization = Anonymization.blur,
    ) -> CameraFrame:
        camera_name = f"{camera.value}_{anonymization.value}"
        return min(
            self.camera_frames[camera_name],
            key=lambda camera_frame: abs(camera_frame.time - time),
        )

    def get_lidar_frame(self, time: datetime, lidar: Lidar = Lidar.velodyne) -> SensorFrame:
        return min(
            self.lidar_frames[lidar.value],
            key=lambda lidar_frame: abs(lidar_frame.time - time),
        )

    ### Full accessors ###

    def get_annotations(self, project: AnnotationProject) -> List[AnnotationFrame]:
        if project.value not in self.annotation_frames:
            return []
        return self.annotation_frames[project.value]

    def get_camera_frames(
        self,
        camera: Camera = Camera.front,
        anonymization: Anonymization = Anonymization.blur,
    ) -> List[CameraFrame]:
        camera_name = f"{camera.value}_{anonymization.value}"
        return self.camera_frames[camera_name]

    def get_lidar_frames(self, lidar: Lidar = Lidar.velodyne) -> List[SensorFrame]:
        return self.lidar_frames[lidar.value]

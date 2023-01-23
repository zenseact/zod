import os.path as osp
from dataclasses import dataclass
from datetime import datetime
from itertools import chain
from typing import Dict, Iterator, List, Tuple

from zod.constants import AnnotationProject, Anonymization, Camera, Lidar

from ._serializable import JSONSerializable
from .sensor import (
    AnnotationFrame,
    CameraFrame,
    LidarFrame,
    SensorFrame,
)


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

    annotation_frames: Dict[AnnotationProject, List[AnnotationFrame]]
    camera_frames: Dict[str, List[CameraFrame]]  # key is a combination of Camera and Anonymization
    lidar_frames: Dict[Lidar, List[LidarFrame]]

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
        self, anonymization: Anonymization, camera: Camera, lidar: Lidar
    ) -> Iterator[Tuple[CameraFrame, SensorFrame]]:
        """Iterate over all camera frames and their corresponding lidar frames.

        Args:
            camera: The camera to use. e.g., camera_front_blur
            lidar: The lidar to use. e.g., lidar_velodyne
        Yields:
            A tuple of the camera frame and the closest lidar frame.
        """
        camera_name = f"{camera.value}_{anonymization.value}"
        assert (
            camera_name in self.camera_frames
        ), f"Camera {camera_name} not found. Available cameras: {self.camera_frames.keys()}"
        assert (
            lidar in self.lidar_frames
        ), f"Lidar {lidar} not found. Available lidars: {self.lidar_frames.keys()}"

        for camera_frame in self.camera_frames[camera_name]:
            # Get the closest lidar frame in time
            lidar_frame = min(
                self.lidar_frames[lidar],
                key=lambda lidar_frame: abs(lidar_frame.time - camera_frame.time),
            )
            yield camera_frame, lidar_frame

    ### Keyframe accessors ###

    def get_key_annotation_frame(self, project: AnnotationProject) -> AnnotationFrame:
        if project not in self.annotation_frames:
            raise ValueError(f"No annotations found for project {project.value}.")
        if len(self.annotation_frames[project]) == 1:
            return self.annotation_frames[project][0]
        else:
            return self.get_annotation_frame(self.keyframe_time, project)

    def get_key_camera_frame(
        self,
        anonymization: Anonymization,
        camera: Camera = Camera.FRONT,
    ) -> CameraFrame:
        camera_name = f"{camera.value}_{anonymization.value}"
        if len(self.camera_frames[camera_name]) == 1:
            return self.camera_frames[camera_name][0]
        else:
            return min(
                self.camera_frames[camera_name],
                key=lambda camera_frame: abs(camera_frame.time - self.keyframe_time),
            )

    def get_key_lidar_frame(self, lidar: Lidar = Lidar.VELODYNE) -> LidarFrame:
        return self.get_lidar_frame(self.keyframe_time, lidar)

    ### Timestamp accessors ###

    def get_annotation_frame(
        self,
        time: datetime,
        project: AnnotationProject,
        # TODO: implement exact flag?
    ) -> AnnotationFrame:
        if project not in self.annotation_frames:
            raise ValueError(f"No annotations found for project {project.value}.")
        return min(
            self.annotation_frames[project],
            key=lambda annotation_frame: abs(annotation_frame.time - time),
        )

    def get_camera_frame(
        self,
        time: datetime,
        anonymization: Anonymization,
        camera: Camera = Camera.FRONT,
    ) -> CameraFrame:
        camera_name = f"{camera.value}_{anonymization.value}"
        return min(
            self.camera_frames[camera_name],
            key=lambda camera_frame: abs(camera_frame.time - time),
        )

    def get_lidar_frame(self, time: datetime, lidar: Lidar = Lidar.VELODYNE) -> LidarFrame:
        return min(
            self.lidar_frames[lidar],
            key=lambda lidar_frame: abs(lidar_frame.time - time),
        )

    ### Full accessors ###

    def get_annotation_frames(self, project: AnnotationProject) -> List[AnnotationFrame]:
        if project not in self.annotation_frames:
            return []
        return self.annotation_frames[project]

    def get_camera_frames(
        self,
        anonymization: Anonymization,
        camera: Camera = Camera.FRONT,
    ) -> List[CameraFrame]:
        camera_name = f"{camera.value}_{anonymization.value}"
        return self.camera_frames[camera_name]

    def get_lidar_frames(self, lidar: Lidar = Lidar.VELODYNE) -> List[LidarFrame]:
        return self.lidar_frames[lidar]

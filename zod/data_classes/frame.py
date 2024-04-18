from typing import Any, List, Optional

import numpy as np

from zod.constants import AnnotationProject, Anonymization, Camera, Lidar, NotAnnotatedError
from zod.utils.compensation import motion_compensate_scanwise

from .calibration import Calibration
from .ego_motion import EgoMotion
from .info import Information
from .metadata import FrameMetaData
from .sensor import CameraFrame, LidarData, LidarFrame


class ZodFrame:
    def __init__(self, info: Information):
        self.info: Information = info  # Holds all the paths to the files
        self._ego_motion: Optional[EgoMotion] = None  # This is the lightweight version of oxts
        self._oxts: Optional[EgoMotion] = None
        self._calibration: Optional[Calibration] = None
        self._metadata: Optional[FrameMetaData] = None

    @property
    def ego_motion(self) -> EgoMotion:
        """Get the ego motion."""
        if self._ego_motion is None:
            self._ego_motion = EgoMotion.from_json_path(self.info.ego_motion_path)
        return self._ego_motion

    @property
    def oxts(self) -> EgoMotion:
        """Get the oxts."""
        if self._oxts is None:
            self._oxts = EgoMotion.from_oxts_path(self.info.oxts_path)
        return self._oxts

    @property
    def calibration(self) -> Calibration:
        """Get the calibration."""
        if self._calibration is None:
            self._calibration = Calibration.from_json_path(self.info.calibration_path)
        return self._calibration

    @property
    def metadata(self) -> FrameMetaData:
        """Get the metadata."""
        if self._metadata is None:
            self._metadata = FrameMetaData.from_json_path(self.info.metadata_path)
        return self._metadata

    def is_annotated(self, project: AnnotationProject) -> bool:
        """Check if the frame is annotated for a given project."""
        return project in self.info.annotations

    def get_annotation(self, project: AnnotationProject) -> List[Any]:
        """Get the annotation for a given project."""
        if not self.is_annotated(project):
            raise NotAnnotatedError(f"Project {project} is not annotated for frame {self.info.id}.")
        return self.info.annotations[project].read()

    def get_camera_frame(self, anonymization: Anonymization = Anonymization.BLUR) -> CameraFrame:
        """Get the camera frame."""
        return self.info.get_key_camera_frame(camera=Camera.FRONT, anonymization=anonymization)

    def get_image(
        self,
        anonymization: Anonymization = Anonymization.BLUR,
    ) -> np.ndarray:
        """Get the image."""
        return self.info.get_key_camera_frame(camera=Camera.FRONT, anonymization=anonymization).read()

    def get_lidar_frames(
        self,
        num_before: int = 0,
        num_after: int = 0,
    ) -> List[LidarFrame]:
        """Get the lidar frames (around the keyframe)."""
        all_frames = self.info.get_lidar_frames(Lidar.VELODYNE)
        key_frame_dx = len(all_frames) // 2  # the key frame is in the middle
        start = max(0, key_frame_dx - num_before)
        end = min(len(all_frames), key_frame_dx + num_after + 1)
        return all_frames[start:end]

    def get_lidar(self, num_before: int = 0, num_after: int = 0) -> List[LidarData]:
        """Get the lidar data, same as `get_lidar_frames` but actually reads the data."""
        return [lidar_frame.read() for lidar_frame in self.get_lidar_frames(num_before, num_after)]

    def compensate_lidar(self, data: LidarData, timestamp: float) -> LidarData:
        """Compensate a point cloud to a given timestamp."""
        lidar_calib = self.calibration.lidars[Lidar.VELODYNE]
        return motion_compensate_scanwise(data, self.ego_motion, lidar_calib, timestamp)

    def get_aggregated_lidar(self, num_before: int, num_after: int = 0, timestamp: Optional[float] = None) -> LidarData:
        """Get an aggregated point cloud around the keyframe."""
        key_lidar_frame = self.info.get_key_lidar_frame()
        key_lidar_data = key_lidar_frame.read()
        _adjust_lidar_core_time(key_lidar_data)
        if timestamp is None:
            timestamp = key_lidar_data.core_timestamp
        lidar_calib = self.calibration.lidars[Lidar.VELODYNE]
        # Adjust each individual scan
        to_aggregate = []
        for lidar_frame in self.get_lidar_frames(num_before, num_after):
            if lidar_frame == key_lidar_frame:
                continue
            lidar_data = lidar_frame.read()
            _adjust_lidar_core_time(lidar_data)
            lidar_data = motion_compensate_scanwise(lidar_data, self.ego_motion, lidar_calib, timestamp)
            to_aggregate.append(lidar_data)
        # Aggregate the scans
        key_lidar_data.extend(*to_aggregate)
        return key_lidar_data


def _adjust_lidar_core_time(lidar: LidarData):
    """Adjust the core timestamp of a lidar scan so that it always points "forward".

    This assumes that the scans are not pointwise compensated, and that the cut-off is to the right.
    """
    # TODO: maybe this should be done by looking at the angles instead?
    # This could fail if there are no points on either side of the cut-off.
    lidar.core_timestamp = 0.75 * lidar.timestamps.max() + 0.25 * lidar.timestamps.min()

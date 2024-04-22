from datetime import datetime
from typing import Any, List, Optional

from zod.constants import AnnotationProject, Lidar, NotAnnotatedError
from zod.utils.compensation import motion_compensate_scanwise

from .calibration import Calibration
from .ego_motion import EgoMotion
from .info import Information
from .metadata import SequenceMetadata
from .sensor import LidarData
from .vehicle_data import VehicleData


class ZodSequence:
    def __init__(self, info: Information) -> None:
        self.info: Information = info  # holds all the paths to the files
        self._ego_motion: Optional[EgoMotion] = None  # this is the light-weight version of oxts
        self._oxts: Optional[EgoMotion] = None
        self._calibration: Optional[Calibration] = None
        self._metadata: Optional[SequenceMetadata] = None
        self._vehicle_data: Optional[VehicleData] = None

    @property
    def ego_motion(self) -> EgoMotion:
        """Get the oxts file."""
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
    def metadata(self) -> SequenceMetadata:
        """Get the metadata."""
        if self._metadata is None:
            self._metadata = SequenceMetadata.from_json_path(self.info.metadata_path)
        return self._metadata

    @property
    def vehicle_data(self) -> VehicleData:
        """Get the vehicle data."""
        if self._vehicle_data is None:
            assert self.info.vehicle_data_path is not None, "Vehicle data path is missing."
            self._vehicle_data = VehicleData.from_hdf5(self.info.vehicle_data_path)
        return self._vehicle_data

    def is_annotated(self, project: AnnotationProject) -> bool:
        """Check if the frame is annotated for a given project."""
        return project in self.info.annotations

    def get_annotation(self, project: AnnotationProject) -> List[Any]:
        """Get the annotation for a given project."""
        if not self.is_annotated(project):
            raise NotAnnotatedError(f"Project {project} is not annotated for sequence {self.info.id}.")
        return self.info.annotations[project].read()

    def get_lidar(self, start: int = 0, end: Optional[int] = None) -> List[LidarData]:
        """Get the point clouds."""
        return [lidar_frame.read() for lidar_frame in self.info.get_lidar_frames(Lidar.VELODYNE)[start:end]]

    def get_compensated_lidar(self, time: datetime) -> LidarData:
        """Get the point cloud at a given timestamp."""
        lid_frame = self.info.get_lidar_frame(time, Lidar.VELODYNE)
        pcd = lid_frame.read()
        return motion_compensate_scanwise(
            pcd,
            self.ego_motion,
            self.calibration.lidars[Lidar.VELODYNE],
            time.timestamp(),
        )

    def get_aggregated_lidar(self, start: int = 0, end: int = None, timestamp: Optional[float] = None) -> LidarData:
        """Get the aggregated point cloud."""
        lidar_scans = self.get_lidar(start, end)
        if timestamp is None:
            timestamp = lidar_scans[len(lidar_scans) // 2].core_timestamp
        for i, scan in enumerate(lidar_scans):
            if scan.core_timestamp == timestamp:
                continue
            compensated_scan = motion_compensate_scanwise(
                scan, self.ego_motion, self.calibration.lidars[Lidar.VELODYNE], timestamp
            )
            lidar_scans[i] = compensated_scan
        aggregated = lidar_scans[0]
        aggregated.extend(*lidar_scans[1:])
        return aggregated

    def get_keyframe_lidar(self, motion_compensated=True) -> LidarData:
        """Get the keyframe point cloud.

        Args:
            motion_compensated (bool, optional): Whether to motion compensate the point cloud
             to camera (and annotation) timestamp. Defaults to True.

        Returns:
            LidarData: The point cloud.
        """

        lidar_scan = self.info.get_key_lidar_frame(Lidar.VELODYNE).read()

        if not motion_compensated:
            return lidar_scan

        return motion_compensate_scanwise(
            lidar_scan,
            self.ego_motion,
            self.calibration.lidars[Lidar.VELODYNE],
            self.info.keyframe_time.timestamp(),
        )

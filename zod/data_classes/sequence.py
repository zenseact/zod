from typing import Any, List, Optional

from zod.constants import AnnotationProject, Lidar
from zod.utils.compensation import motion_compensate_scanwise

from .calibration import Calibration
from .info import Information
from .metadata import SequenceMetadata
from .oxts import EgoMotion
from .sensor import LidarData


class ZodSequence:
    def __init__(self, info: Information):
        self.info: Information = info  # holds all the paths to the files
        self._ego_motion: EgoMotion = None  # this is the light-weight version of oxts
        self._oxts: EgoMotion = None
        self._calibration: Calibration = None
        self._metadata: SequenceMetadata = None

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

    def get_annotation(self, project: AnnotationProject) -> List[Any]:
        """Get the annotation for a given project."""
        anno_frame = self.info.get_key_annotation_frame(project)
        return anno_frame and anno_frame.read()  # read if not None

    def get_point_clouds(self, start: int = 0, end: int = -1) -> List[LidarData]:
        """Get the point clouds."""
        return [
            lidar_frame.read()
            for lidar_frame in self.info.get_lidar_frames(Lidar.VELODYNE)[start:end]
        ]

    def get_aggregated_point_cloud(
        self, start: int = 0, end: int = -1, timestamp: Optional[float] = None
    ) -> LidarData:
        """Get the aggregated point cloud."""
        lidar_scans = self.get_point_clouds(start, end)
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

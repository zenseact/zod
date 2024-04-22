"""Top-level package for Zenseact Open Dataset (ZOD)."""

import importlib.metadata as importlib_metadata

from ._zod_dataset import ZodDataset as ZodDataset  # For type hinting
from .anno.ego_road import EgoRoadAnnotation as EgoRoadAnnotation
from .anno.lane import LaneAnnotation as LaneAnnotation
from .anno.object import ObjectAnnotation as ObjectAnnotation
from .anno.road_condition import RoadConditionAnnotation as RoadConditionAnnotation
from .anno.tsr.traffic_sign import TrafficSignAnnotation as TrafficSignAnnotation
from .constants import AnnotationProject as AnnotationProject
from .constants import Anonymization as Anonymization
from .constants import Camera as Camera
from .constants import Lidar as Lidar
from .data_classes.frame import ZodFrame as ZodFrame
from .data_classes.sequence import ZodSequence as ZodSequence
from .zod_drives import ZodDrives as ZodDrives
from .zod_frames import ZodFrames as ZodFrames
from .zod_sequences import ZodSequences as ZodSequences

try:
    __version__ = importlib_metadata.version(__package__ or __name__)
except importlib_metadata.PackageNotFoundError:
    __version__ = "0.0.0"

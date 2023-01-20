"""Relevant constants for ZOD."""

from enum import Enum
from typing import Literal, Union

# Dataset paths
SINGLE_FRAMES = "single_frames"
SEQUENCES = "sequences"
DRIVES = "drives"
DB_DATE_STRING_FORMAT_W_MICROSECONDS = "%Y-%m-%dT%H:%M:%S.%f%z"

# Train-val splits
TRAIN = "train"
VAL = "val"
FULL = "full"
MINI = "mini"
SINGLE = "single"
SPLIT_TO_TRAIN_VAL_FILE_SINGLE_FRAMES = {
    FULL: f"{FULL}_train_val_{SINGLE_FRAMES}.json",
    MINI: f"{MINI}_train_val_{SINGLE_FRAMES}.json",
}
SPLIT_TO_TRAIN_VAL_FILE_SEQUENCES = {
    FULL: f"{FULL}_train_val_{SEQUENCES}.json",
    MINI: f"{MINI}_train_val_{SEQUENCES}.json",
}

VERSIONS = (FULL, MINI)


class Anonymization(Enum):
    BLUR = "blur"
    DNAT = "dnat"
    ORIGINAL = "original"


class AnnotationProject(Enum):
    # TODO: delete this!
    DYNAMIC_OBJECTS = "dynamic_objects"
    STATIC_OBJECTS = "static_objects"
    OBJECT_DETECTION = "object_detection"
    LANE_MARKINGS = "lane_markings"
    TRAFFIC_SIGNS = "traffic_signs"
    EGO_ROAD = "ego_road"
    ROAD_CONDITION = "road_condition"


### Coordinate Frames ###

EGO = "ego"


class Camera(Enum):
    FRONT = "front"


class Lidar(Enum):
    VELODYNE = "velodyne"


CoordinateFrame = Union[Camera, Lidar, Literal[EGO]]


### Evaluation ###

EVALUATION_FRAME = Lidar.VELODYNE
ALL_CLASSES = [
    "Vehicle",
    "VulnerableVehicle",
    "Pedestrian",
    "Animal",
    "PoleObject",
    "Inconclusive",
    "TrafficBeacon",
    "TrafficSign",
    "TrafficSignal",
    "TrafficGuide",
    "DynamicBarrier",
]

EVALUATION_CLASSES = [
    "Vehicle",
    "VulnerableVehicle",
    "Pedestrian",
    "TrafficSign",
    "TrafficSignal",
]

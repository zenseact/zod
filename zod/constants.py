"""Relevant constants for ZOD."""

from enum import Enum
from typing import Literal, Union

# Dataset paths
FRAMES = "single_frames"
SEQUENCES = "sequences"
DRIVES = "drives"
DB_DATE_STRING_FORMAT_W_MICROSECONDS = "%Y-%m-%dT%H:%M:%S.%f%z"

# Train-val splits
TRAIN = "train"
VAL = "val"
FULL = "full"
MINI = "mini"
TRAINVAL_FILES = {
    FRAMES: {
        FULL: f"trainval_frames_full.json",
        MINI: f"trainval_frames_mini.json",
    },
    SEQUENCES: {
        FULL: f"trainval_sequences_full.json",
        MINI: f"trainval_sequences_mini.json",
    },
}
SPLIT_FILES = {
    SEQUENCES: {
        FULL: {
            TRAIN: "splits/sequences_full_train.txt",
            VAL: "splits/sequences_full_val.txt",
        },
        MINI: {
            TRAIN: "splits/sequences_mini_train.txt",
            VAL: "splits/sequences_mini_val.txt",
        },
    },
    FRAMES: {
        FULL: {
            TRAIN: "splits/frames_full_train.txt",
            VAL: "splits/frames_full_val.txt",
        },
        MINI: {
            TRAIN: "splits/frames_mini_train.txt",
            VAL: "splits/frames_mini_val.txt",
        },
    },
}


VERSIONS = (FULL, MINI)


class Anonymization(Enum):
    BLUR = "blur"
    DNAT = "dnat"
    ORIGINAL = "original"


class AnnotationProject(Enum):
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

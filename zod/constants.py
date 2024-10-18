"""Relevant constants for ZOD."""

import typing
from enum import Enum
from typing import Literal, Union

# Dataset paths
FRAMES = "single_frames"
SEQUENCES = "sequences"
DRIVES = "drives"
DB_DATE_STRING_FORMAT_W_MICROSECONDS = "%Y-%m-%dT%H:%M:%S.%f%z"

Split = Literal["train", "val", "blacklisted"]
SPLITS = typing.get_args(Split)
TRAIN, VAL, BLACKLISTED = SPLITS

Version = Literal["full", "mini"]
VERSIONS = typing.get_args(Version)
FULL, MINI = VERSIONS

TRAINVAL_FILES = {
    FRAMES: {
        FULL: f"trainval-frames-full.json",
        MINI: f"trainval-frames-mini.json",
    },
    SEQUENCES: {
        FULL: f"trainval-sequences-full.json",
        MINI: f"trainval-sequences-mini.json",
    },
    DRIVES: {
        FULL: f"trainval-drives-full.json",
        MINI: f"trainval-drives-mini.json",
    },
}
SPLIT_FILES = {
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
    DRIVES: {
        FULL: {
            TRAIN: "splits/drives_full_train.txt",
            VAL: "splits/drives_full_val.txt",
        },
        MINI: {
            TRAIN: "splits/drives_mini_train.txt",
            VAL: "splits/drives_mini_val.txt",
        },
    },
}


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


class NotAnnotatedError(Exception):
    """Error that is raised when a project is not annotated for the requested id."""


### Coordinate Frames ###


class Camera(Enum):
    FRONT = "front"


class Lidar(Enum):
    VELODYNE = "velodyne"


class Radar(Enum):
    FRONT = "front"


Ego = Literal["ego"]
EGO = typing.get_args(Ego)[0]
CoordinateFrame = Union[Camera, Lidar, Radar, Ego]


# ZodFrame properties
FRAMES_IMAGE_MEAN = [0.337, 0.345, 0.367]
FRAMES_IMAGE_STD = [0.160, 0.180, 0.214]

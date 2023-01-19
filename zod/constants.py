"""Relevant constants for ZOD."""

from enum import Enum


# Sensor data
LIDAR_VELODYNE = "lidar_velodyne"
CAMERA_FRONT = "camera_front"
BLUR = "blur"
DNAT = "dnat"
CAMERA_FRONT_BLUR = f"{CAMERA_FRONT}_{BLUR}"
CAMERA_FRONT_DNAT = f"{CAMERA_FRONT}_{DNAT}"
CALIBRATION = "calibration"
OXTS = "oxts"
LIDARS = (LIDAR_VELODYNE,)
CAMERAS = (CAMERA_FRONT,)
EGO = "ego"

# Evaluation
EVALUATION_FRAME = LIDAR_VELODYNE
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

# Keys in oxts data
LONGITUDE = "posLon"
LATITUDE = "posLat"
MAPS_STYLE = "open-street-map"
SIZE_MAX = 7
OPACITY_LEVEL = 1
DEFAULT_COLOR = "red"
DEFAULT_SIZE = 1

# Useful constants
MICROSEC_PER_SEC = 1e6


class Camera(Enum):
    front = "front"


class Lidar(Enum):
    velodyne = "velodyne"


class Anonymization(Enum):
    blur = BLUR
    dnat = DNAT
    original = "original"


class AnnotationProject(Enum):
    OBJECT_DETECTION = "object_detection"
    LANE_MARKINGS = "lane_markings"
    TRAFFIC_SIGNS = "traffic_signs"
    EGO_ROAD = "ego_road"
    ROAD_CONDITION = "road_condition"

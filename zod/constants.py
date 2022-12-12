"""Relevant constants for the ZOD."""

# sensor data
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

# evaluatuion
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

# dataset paths
SINGLE_FRAMES = "single_frames"

# train-val splits
TRAIN = "train"
VAL = "val"
FULL = "full"
MINI = "mini"
SINGLE = "single"
SPLIT_TO_TRAIN_VAL_FILE_SINGLE_FRAMES = {
    FULL: f"{FULL}_train_val_{SINGLE_FRAMES}.json",
    MINI: f"{MINI}_train_val_{SINGLE_FRAMES}.json",
}
VERSIONS = (FULL, MINI)

# keys in OxTS data
LONGITUDE = "posLon"
LATITUDE = "posLat"
MAPS_STYLE = "open-street-map"
SIZE_MAX = 7
OPACITY_LEVEL = 1
DEFAULT_COLOR = "red"
DEFAULT_SIZE = 1

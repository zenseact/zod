import numpy as np
import h5py

from dataclasses import dataclass


@dataclass
class CornerRadarData:
    name: str
    header_timestamps: np.ndarray
    lat_mounting: float  # meters
    lon_mounting: float  # meters
    angle_mounting: float  # radians

    def write_to_file(self, output_file: str):
        pass


def _process_radar_data(file: h5py.File, name: str) -> CornerRadarData:
    data = file[f"radardata_{name}"]["data"]
    timestamps_ns = np.array(file[f"radardata_{name}"]["zeader"]["timestamp_ns"])

    lat_mounting = data["mounting_position"]["lat_pos"]["meters"]["value"][0]
    lon_mounting = data["mounting_position"]["long_pos"]["meters"]["value"][0]
    angle_mounting = data["mounting_position"]["mounting_angle"]["radians"]["value"][0]

    return CornerRadarData(
        name,
        timestamps_ns,
        lat_mounting,
        lon_mounting,
        angle_mounting,
    )


def process_radar_data_fsrl(file: h5py.File) -> CornerRadarData:
    return _process_radar_data(file, "fsrl")


def process_radar_data_fsrr(file: h5py.File) -> CornerRadarData:
    return _process_radar_data(file, "fsrr")


def process_radar_data_rsrl(file: h5py.File) -> CornerRadarData:
    return _process_radar_data(file, "rsrl")


def process_radar_data_rsrr(file: h5py.File) -> CornerRadarData:
    return _process_radar_data(file, "rsrr")


def process_radar_data_flr(file: h5py.File) -> CornerRadarData:
    return _process_radar_data(file, "flr")

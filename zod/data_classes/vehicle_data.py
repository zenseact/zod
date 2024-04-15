import os.path as osp
from dataclasses import dataclass
from typing import Any, Dict

import h5py
import numpy as np


@dataclass
class EgoVehicleData:
    roll_rate: np.ndarray  # radians/s
    picth_rate: np.ndarray  # radians/s
    lat_vel: np.ndarray  # m/s
    lon_vel: np.ndarray  # m/s
    lat_acc: np.ndarray  # m/s^2
    lon_acc: np.ndarray  # m/s^2
    body_height: np.ndarray  # meters
    body_pitch: np.ndarray  # radians
    timestamp: np.ndarray  # epoch time, nanosec

    @classmethod
    def from_hdf5(cls, path: str) -> "EgoVehicleData":
        """Read the ego-vehicle data from the hdf5 file."""
        assert osp.exists(path), "Please download the vehicle data first."
        with h5py.File(path, "r") as h5_file:
            evd = h5_file["ego_vehicle_data"]
            return cls(
                roll_rate=evd["roll_rate_data/angle_rate/radians_per_second/value"][:],
                picth_rate=evd["pitch_rate_data/angle_rate/radians_per_second/value"][:],
                lat_vel=evd["lat_vel_data/velocity/meters_per_second/value"][:],
                lon_vel=evd["lon_vel_data/velocity/meters_per_second/value"][:],
                lat_acc=evd["lat_acc_data/acceleration/meters_per_second2/value"][:],
                lon_acc=evd["lon_acc_data/acceleration/meters_per_second2/value"][:],
                body_height=evd["body_height/body_height/meters/value"][:],
                body_pitch=evd["body_pitch/angle/radians/value"][:],
                timestamp=evd["timestamp/nanoseconds/value"][:],
            )


@dataclass
class EgoVehicleControls:
    acc_pedal: np.ndarray  # ratio, range 0-100
    brake_pedal_pressed: np.ndarray  # bool
    steering_angle: np.ndarray  # radians, counter-clockwise positive
    steering_angle_rate: np.ndarray  # radians/s
    steering_wheel_torque: np.ndarray  # Nm
    turn_indicator: np.ndarray  # 0: off, 1: left, 2: right
    timestamp: np.ndarray  # epoch time, nanosec

    @classmethod
    def from_hdf5(cls, path: str) -> "EgoVehicleControls":
        """Read the ego-vehicle control data from the hdf5 file."""
        assert osp.exists(path), "Please download the vehicle data first."
        with h5py.File(path, "r") as h5_file:
            evc = h5_file["ego_vehicle_controls"]
            return cls(
                acc_pedal=evc["acceleration_pedal/ratio/unitless/value"][:],
                brake_pedal_pressed=evc["brake_pedal_pressed/is_brake_pedal_pressed/unitless/value"][:],
                steering_angle=evc["steering_wheel_angle/angle/radians/value"][:],
                steering_angle_rate=evc["steering_wheel_angle/angle_rate/radians_per_second/value"][:],
                steering_wheel_torque=evc["steer_wheel_torque/torque/newton_meters/value"][:],
                turn_indicator=evc["turn_indicator_status/state"][:],
                timestamp=evc["timestamp/nanoseconds/value"][:],
            )


@dataclass
class Satellite:
    altitude: np.ndarray  # meters
    heading: np.ndarray  # degrees
    latpos: np.ndarray  # nano degrees
    lonpos: np.ndarray  # nano degrees
    nrof_satellites: np.ndarray  # number of satellites, integer
    speed: np.ndarray  # m/s
    timstamp: np.ndarray  # epoch time, nanosec

    @classmethod
    def from_hdf5(cls, path: str) -> "Satellite":
        """Read the sattelite data from the hdf5 file."""
        assert osp.exists(path), "Please download the vehicle data first."
        with h5py.File(path, "r") as h5_file:
            s = h5_file["satellite"]
            return cls(
                altitude=s["altitude/meters/value"][:],
                heading=s["heading/degrees/value"][:],
                latpos=s["latposn/nanodegrees/value"][:],
                lonpos=s["longposn/nanodegrees/value"][:],
                nrof_satellites=s["nrof_satellites/unitless/value"][:],
                speed=s["speed/meters_per_second/value"][:],
                timstamp=s["timestamp/nanoseconds/value"][:],
            )


@dataclass
class VehicleData:
    """Class to store the vehicle data."""

    ego_vehicle_data: EgoVehicleData
    ego_vehicle_controls: EgoVehicleControls
    satellite: Satellite

    @classmethod
    def from_hdf5(cls, path: str) -> "VehicleData":
        """Read the vehicle data from the hdf5 file."""
        assert osp.exists(path), "Please download the vehicle data first."
        return cls(
            EgoVehicleData.from_hdf5(path),
            ego_vehicle_controls=EgoVehicleControls.from_hdf5(path),
            satellite=Satellite.from_hdf5(path),
        )

import os.path as osp
from dataclasses import dataclass
from typing import Any, Dict

import h5py


@dataclass
class VehicleData:
    """Class to store the vehicle data."""

    ego_vehicle_data: Dict[str, Any]
    ego_vehicle_controls: Dict[str, Any]
    satellite: Dict[str, Any]

    @classmethod
    def from_hdf5(cls, path: str) -> "VehicleData":
        """Read the vehicle data from the hdf5 file."""
        assert osp.exists(path), "Please download the vehicle data first."
        with h5py.File(path, "r") as h5_file:
            return cls(
                ego_vehicle_data=_load_group_data(h5_file, "ego_vehicle_data"),
                ego_vehicle_controls=_load_group_data(h5_file, "ego_vehicle_controls"),
                satellite=_load_group_data(h5_file, "satellite"),
            )


def _load_group_data(h5_file, group_name) -> Dict[str, Any]:
    group_data = {}
    for key in h5_file[group_name].keys():
        group_data[key] = h5_file[group_name][key][()]
    return group_data

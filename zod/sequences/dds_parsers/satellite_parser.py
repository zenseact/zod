import h5py
from dataclasses import dataclass


@dataclass
class SatelliteData:
    pass

    def write_to_file(self, output_file: str):
        pass


def process_satellite(file: h5py.File) -> SatelliteData:
    pass

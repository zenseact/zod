from dataclasses import dataclass
from datetime import datetime

from ._serializable import JSONSerializable


@dataclass
class FrameMetaData(JSONSerializable):
    """A class describing the metadata of a frame."""

    frame_id: str
    time: datetime
    country_code: str
    scraped_weather: str
    collection_car: str
    road_type: str
    road_condition: str
    time_of_day: str
    num_lane_instances: int
    num_vehicles: int
    num_vulnerable_vehicles: int
    num_pedestrians: int
    num_traffic_lights: int
    num_traffic_signs: int
    longitude: float
    latitude: float
    solar_angle_elevation: float


@dataclass
class SequenceMetadata(JSONSerializable):
    """A class describing the metadata of a sequence."""

    sequence_id: str
    start_time: datetime
    end_time: datetime
    country_code: str
    collection_car: str
    longitude: float
    latitude: float

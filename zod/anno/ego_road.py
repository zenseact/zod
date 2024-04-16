from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List


class EgoRoadClass(Enum):
    ROAD = "EgoRoad_Road"
    DEBRIS = "EgoRoad_Debris"


@dataclass
class EgoRoadAnnotation:
    """EgoRoad dataclass."""

    uuid: str
    geometry: List[List[float]]  # [[x1, y1], [x2, y2], ...]
    type: EgoRoadClass

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EgoRoadAnnotation:
        return cls(
            uuid=data["properties"]["annotation_uuid"],
            geometry=data["geometry"]["coordinates"],
            type=EgoRoadClass(data["properties"]["class"]),
        )

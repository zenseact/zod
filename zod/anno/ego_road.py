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
    _type: EgoRoadClass

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EgoRoadAnnotation:
        properties: Dict[str, Any] = data["properties"]
        return cls(
            uuid=properties["annotation_uuid"],
            geometry=data["geometry"]["coordinates"],
            type=EgoRoadClass(properties["class"]),
        )

    @property
    def is_road(self) -> bool:
        """Return True if the annotation is a road."""
        return self._type == EgoRoadClass.ROAD

    @property
    def is_debris(self) -> bool:
        """Return True if the annotation is debris."""
        return self._type == EgoRoadClass.DEBRIS

    @property
    def type(self) -> EgoRoadClass:
        """Return the type of thepru annotation."""
        return self._type

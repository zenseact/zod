from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

LANE_MARKING_TYPES = []  # TODO


@dataclass
class LaneAnnotation(ABC):
    """Base class for lane annotations."""

    uuid: str
    geometry: List[List[float]]  # [[x1, y1], [x2, y2], ...]


@dataclass
class LaneMarkingAnnotation(LaneAnnotation):
    """Class to store lane marking information."""

    type: str  # e.g. "solid" or "dashed"
    colored: bool
    instance_id: Optional[int]
    cardinality: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LaneMarkingAnnotation:
        """Create an AnnotatedLaneMarking from a dictionary."""
        properties: Dict[str, Any] = data["properties"]
        if "InstanceID" not in properties or "MultipleLaneMarkings" not in properties:
            pass
        return cls(
            uuid=properties["annotation_uuid"],
            geometry=data["geometry"]["coordinates"],
            type=properties["class"],
            colored=properties["coloured"],
            instance_id=properties.get("InstanceID"),
            cardinality=properties.get("MultipleLaneMarkings", "Single"),
        )


@dataclass
class ShadedAreaAnnotation(LaneAnnotation):
    """Class to store shaded area information."""

    type: str
    colored: bool

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ShadedAreaAnnotation:
        """Create an AnnotatedLaneMarking from a dictionary."""
        properties: Dict[str, Any] = data["properties"]
        return cls(
            uuid=properties["annotation_uuid"],
            geometry=data["geometry"]["coordinates"],
            type=properties["ShadedAreaType"],
            colored=properties["coloured"],
        )


@dataclass
class RoadPaintingAnnotation(LaneAnnotation):
    """Class to store road painting information."""

    unclear: Optional[bool]
    odd: Optional[bool]
    contains_arrow: Optional[bool]
    contains_pictogram: Optional[bool]
    contains_text: Optional[bool]
    contains_trafficsign: Optional[bool]
    contains_crosswalk: Optional[bool]
    contains_marker: Optional[bool]
    contains_other: Optional[bool]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RoadPaintingAnnotation:
        """Create an AnnotatedRoadPainting from a dictionary."""
        properties: Dict[str, Any] = data["properties"]

        return cls(
            uuid=properties["annotation_uuid"],
            geometry=data["geometry"]["coordinates"],
            unclear=properties.get("Unclear", None),
            odd=properties.get("Odd", None),
            contains_arrow=properties.get("ContainsArrow", None),
            contains_pictogram=properties.get("ContainsPictogram", None),
            contains_text=properties.get("ContainsText", None),
            contains_trafficsign=properties.get("ContainsTrafficSign", None),
            contains_crosswalk=properties.get("ContainsCrossWalk", None),
            contains_marker=properties.get("ContainsMarker", None),
            contains_other=properties.get("ContainsOther", None),
        )

    @property
    def type(self) -> str:
        """Return the type of road painting."""
        assert (
            sum(
                [
                    bool(self.contains_arrow),
                    bool(self.contains_pictogram),
                    bool(self.contains_text),
                    bool(self.contains_trafficsign),
                    bool(self.contains_crosswalk),
                    bool(self.contains_marker),
                    bool(self.contains_other),
                ]
            )
            <= 1
        ), f"More than one type of road painting is set: {self}"
        if self.contains_arrow:
            return "arrow"
        elif self.contains_pictogram:
            return "pictogram"
        elif self.contains_text:
            return "text"
        elif self.contains_trafficsign:
            return "trafficsign"
        elif self.contains_crosswalk:
            return "crosswalk"
        elif self.contains_marker:
            return "marker"
        elif self.contains_other:
            return "other"
        elif self.unclear:
            print("no road painting type set, but unclear is set", flush=True)
            return "unclear"
        elif self.odd:
            print("no road painting type set, but odd is set", flush=True)
            return "odd"
        else:
            print("ERROR!", flush=True)
            raise ValueError(f"no road painting type set: {self}")


def parse_lane_annotation(data: Dict) -> LaneAnnotation:
    """Parse a lane annotation from a dictionary."""
    if "class" in data["properties"]:
        if data["properties"]["class"] == "shaded_area":
            return ShadedAreaAnnotation.from_dict(data)
        else:
            return LaneMarkingAnnotation.from_dict(data)
    return RoadPaintingAnnotation.from_dict(data)

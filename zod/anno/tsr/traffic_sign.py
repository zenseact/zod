from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from zod.anno.tsr.class_map import get_class_idx
from zod.constants import Camera
from zod.data_classes.box import Box2D


@dataclass
class TrafficSignAnnotation:
    """Traffic sign dataclass."""

    bounding_box: Box2D
    unclear: bool
    uuid: str

    # These are not set if the object is unclear
    traffic_sign_class: Optional[str] = None
    traffic_sign_idx: Optional[int] = None
    occlusion_ratio: Optional[str] = None
    annotation_uuid: Optional[str] = None
    electronic_sign: Optional[bool] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TrafficSignAnnotation:
        """Create a TrafficSignAnnotation from a dict."""
        bbox = Box2D.from_points(data["geometry"]["coordinates"], Camera.FRONT)
        if data["properties"]["class"] == "unclear":
            return cls(
                bounding_box=bbox,
                unclear=True,
                uuid=data["properties"]["annotation_uuid"],
            )
        else:
            return cls(
                unclear=data["properties"]["unclear"],
                bounding_box=Box2D.from_points(data["geometry"]["coordinates"], Camera.FRONT),
                traffic_sign_class=data["properties"]["class"],
                traffic_sign_idx=get_class_idx(data["properties"]["class"]),
                occlusion_ratio=data["properties"]["occlusion_ratio"],
                annotation_uuid=data["properties"]["annotation_uuid"],
                electronic_sign=data["properties"]["is_electronic"],
                uuid=data["properties"]["annotation_uuid"],
            )

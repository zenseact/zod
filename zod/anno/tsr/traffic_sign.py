from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from zod.anno.tsr.class_map import get_class_idx
from zod.constants import Camera
from zod.data_classes.box import Box2D


@dataclass
class TrafficSignAnnotation:
    """Traffic sign dataclass."""

    bounding_box: Box2D
    traffic_sign_class: str
    traffic_sign_idx: int
    occlusion_ratio: str
    annotation_uuid: str
    electronic_sign: bool
    uuid: str
    unclear: bool

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TrafficSignAnnotation:
        """Create a TrafficSignAnnotation from a dict."""
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

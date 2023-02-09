from dataclasses import dataclass

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

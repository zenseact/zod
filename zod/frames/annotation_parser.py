"""Annotation parsers."""
import json
from dataclasses import dataclass
from typing import Any, Dict, List

from zod.constants import CAMERA_FRONT
from zod.frames.traffic_sign_classification.class_map import get_class_idx
from zod.utils.objects import AnnotatedObject, Box2D


def _read_annotation_file(annotation_file: str) -> Dict[str, Any]:
    """Read an annotation file."""
    with open(annotation_file, "r") as file:
        annotation = json.load(file)

    return annotation


def parse_object_detection_annotation(annotation_path: str) -> List[AnnotatedObject]:
    """Parse the objects annotation from the annotation string."""
    annotation = _read_annotation_file(annotation_path)
    return [AnnotatedObject.from_dict(obj) for obj in annotation]


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


def parse_traffic_sign_annotation(annotation_path: str) -> List[TrafficSignAnnotation]:
    """Parse the traffic sign annotation from the annotation string."""
    annotation = _read_annotation_file(annotation_path)
    annotated_objects = []

    for annotated_object in annotation:
        # ignore all unclear traffic signs
        if annotated_object["properties"]["unclear"]:
            continue

        bounding_box = Box2D.from_points(annotated_object["geometry"]["coordinates"], CAMERA_FRONT)

        annotated_objects.append(
            TrafficSignAnnotation(
                bounding_box=bounding_box,
                traffic_sign_class=annotated_object["properties"]["class"],
                traffic_sign_idx=get_class_idx(annotated_object["properties"]["class"]),
                occlusion_ratio=annotated_object["properties"]["occlusion_ratio"],
                annotation_uuid=annotated_object["properties"]["annotation_uuid"],
                electronic_sign=annotated_object["properties"]["is_electronic"],
                uuid=annotated_object["properties"]["annotation_uuid"],
            )
        )

    return annotated_objects


def parse_lane_markings_annotation(annotation_path: str, classes=["lm_dashed", "lm_solid"]):
    """Parse the lane markings annotation from the annotation string."""
    if annotation_path:
        annotations = _read_annotation_file(annotation_path)
    else:
        annotations = None

    polygons = []
    for annotation in annotations:
        if "class" in annotation["properties"]:
            annotated_class = annotation["properties"]["class"]
            if annotated_class in classes:
                polygons.append(annotation["geometry"]["coordinates"])

    return polygons


def parse_ego_road_annotation(annotation_path: str, classes=["EgoRoad_Road"]):
    """Parse the egoroad annotation from the annotation string."""
    if annotation_path:
        annotations = _read_annotation_file(annotation_path)
    else:
        annotations = None

    polygons = []
    for annotation in annotations:
        if "class" in annotation["properties"]:
            annotated_class = annotation["properties"]["class"]
            if annotated_class in classes:
                polygons.append(annotation["geometry"]["coordinates"])

    return polygons


if __name__ == "__main__":
    pass

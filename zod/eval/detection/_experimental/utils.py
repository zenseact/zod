"""Utility functions for object detection evaluation."""

from dataclasses import dataclass
from typing import Union

import numpy as np

# NOTE: we need to add shapely to the requirements once this is enabled
from shapely import geometry

from zod.anno.object import ObjectAnnotation
from zod.data_classes.box import Box2D, Box3D
from zod.eval.detection._nuscenes_eval.detection.data_classes import DetectionBox


@dataclass
class PredictedObject:
    """Class to store dynamic object prediction information."""

    # TODO: maybe move to zod.eval?

    name: str
    confidence: float
    box3d: Box3D

    def __eq__(self, __o: Union[ObjectAnnotation, "PredictedObject"]) -> bool:
        return self.box3d == __o.box3d


def convert_to_detection_box(frame_id: str, obj: ObjectAnnotation) -> DetectionBox:
    """Convert a DynamicObject to a DetectionBox."""
    det_box = DetectionBox(
        sample_token=frame_id,
        translation=tuple(obj.box3d.center),
        size=tuple(obj.box3d.size),
        rotation=tuple(obj.box3d.orientation.elements),
        num_pts=10,
        detection_name=obj.name,
        detection_score=-1.0,  # ground truth boxes have no score
    )

    return det_box


def center_distance(box1: Box3D, box2: Box3D) -> float:
    """Calculate the center distance between two boxes."""
    return np.linalg.norm(box1.center - box2.center)


def center_distance_bev(box1: Box3D, box2: Box3D) -> float:
    """Calculate the center distance between two boxes."""
    assert box1.frame == box2.frame, "Boxes must be in the same frame."
    return np.linalg.norm(box1.center[:2] - box2.center[:2])


def iou3D(box1: Box3D, box2: Box3D) -> float:
    """Calculate the iou between two boxes."""
    assert box1.frame == box2.frame, "Boxes must be in the same frame."

    raise NotImplementedError


def iou2D(box1: Box2D, box2: Box2D) -> float:
    """Calculate the iou between two boxes."""

    # calculate the intersection
    xmin = max(box1.xyxy[0], box2.xyxy[0])
    ymin = max(box1.xyxy[1], box2.xyxy[1])
    xmax = min(box1.xyxy[2], box2.xyxy[2])
    ymax = min(box1.xyxy[3], box2.xyxy[3])
    intersection = max(0, xmax - xmin) * max(0, ymax - ymin)

    # calculate the union
    union = box1.area + box2.area - intersection

    return intersection / union if union > 0.0 else 0.0


def giou2D(box1: Box2D, box2: Box2D) -> float:
    """Calculate the giou between two boxes."""

    # calculate the intersection
    xmin = max(box1.xyxy[0], box2.xyxy[0])
    ymin = max(box1.xyxy[1], box2.xyxy[1])
    xmax = min(box1.xyxy[2], box2.xyxy[2])
    ymax = min(box1.xyxy[3], box2.xyxy[3])
    intersection = max(0, xmax - xmin) * max(0, ymax - ymin)

    # calculate the union
    union = box1.area + box2.area - intersection

    # calculate the convex hull
    xmin = min(box1.xyxy[0], box2.xyxy[0])
    ymin = min(box1.xyxy[1], box2.xyxy[1])
    xmax = max(box1.xyxy[2], box2.xyxy[2])
    ymax = max(box1.xyxy[3], box2.xyxy[3])
    convex_hull = (xmax - xmin) * (ymax - ymin)

    return intersection / union - (convex_hull - union) / convex_hull if union > 0.0 else 0.0


def iod2D(box1: Box2D, detection: Box2D) -> float:
    """Calculate the intersection over detection area between two boxes."""

    # calculate the intersection
    xmin = max(box1.xyxy[0], detection.xyxy[0])
    ymin = max(box1.xyxy[1], detection.xyxy[1])
    xmax = min(box1.xyxy[2], detection.xyxy[2])
    ymax = min(box1.xyxy[3], detection.xyxy[3])
    intersection = max(0, xmax - xmin) * max(0, ymax - ymin)

    # calculate the union
    return intersection / detection.area if detection.area > 0.0 else 0.0


def polygon_iod2D(polygon1: geometry.Polygon, detection: geometry.Polygon) -> float:
    """Calculate the intersection over detection area between two polygons."""
    intersection = polygon1.intersection(detection).area
    return intersection / detection.area if detection.area > 0.0 else 0.0


def polygon_iou2D(polygon1: geometry.Polygon, polygon2: geometry.Polygon) -> float:
    """Calculate the iou between two polygons."""
    intersection = polygon1.intersection(polygon2).area
    union = polygon1.union(polygon2).area
    return intersection / union if union > 0.0 else 0.0


def polygon_giou2D(polygon1: geometry.Polygon, polygon2: geometry.Polygon) -> float:
    """Calculate the giou between two polygons."""
    intersection = polygon1.intersection(polygon2).area
    union = polygon1.union(polygon2).area
    convex_hull = polygon1.convex_hull.union(polygon2.convex_hull).area
    return intersection / union - (convex_hull - union) / convex_hull if union > 0.0 else 0.0

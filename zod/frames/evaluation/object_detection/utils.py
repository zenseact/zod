"""Utility functions for object detection evaluation."""

import numpy as np
from shapely import geometry
from zod.frames.evaluation.object_detection.nuscenes_eval.detection.data_classes import (
    DetectionBox,
)
from zod.utils.objects import Box2D, Box3D, AnnotatedObject

NUSCENES_DEFAULT_SETTINGS = {
    "class_range": {
        "Vehicle": 50,
        "VulnerableVehicle": 40,
        "Pedestrian": 30,
        "TrafficSign": 30,
        "TrafficSignal": 30,
    },
    "dist_fcn": "center_distance",
    "dist_ths": [0.5, 1.0, 2.0, 4.0],
    "dist_th_tp": 2.0,
    "min_recall": 0.1,
    "min_precision": 0.1,
    "max_boxes_per_sample": 500,
    "mean_ap_weight": 5,
}


def convert_to_detection_box(frame_id: str, obj: AnnotatedObject) -> DetectionBox:
    """Convert a DynamicObject to a DetectionBox."""
    det_box = DetectionBox(
        sample_token=frame_id,
        translation=tuple(obj.box3d.center),
        size=tuple(obj.box3d.size),
        rotation=tuple(obj.box3d.orientation.elements),
        velocity=(0, 0),  # we dont have velocity in zod
        ego_translation=tuple(0, 0, 0),  # gt is always at (0, 0, 0)
        num_pts=10,
        detection_name=obj.name,
        detection_score=-1.0,  # ground truth boxes have no score
        attribute_name="",  # we dont have attributes in zod
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

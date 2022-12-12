from typing import List, Sequence, Tuple

import cv2
import numpy as np

from ..utils.objects import AnnotatedObject, Box2D, Box3D
from ..utils.visualization import render_3d_box


FONT_TYPE = cv2.FONT_HERSHEY_COMPLEX


def apply_scale(values: Sequence[float], scale_factor: float) -> Sequence[float]:
    """Apply scale to values."""
    if scale_factor:
        return tuple(map(lambda x: scale_factor * x, values))
    return values


def calc_iou(box1_corners, box2_corners):
    """Calculate IoU between two boxes.

    Args:
        box1_corners (tuple[tuple[float]]) : left-top and right-bottom points of first box
        box2_corners (tuple[tuple[float]]) : left-top and right-bottom points of second box

    Returns:
        iou (float) : IoU metric

    """
    # select inner box corners
    inner_left_coord = max(box1_corners[0][0], box2_corners[0][0])
    inner_top_coord = max(box1_corners[0][1], box2_corners[0][1])
    inner_right_coord = min(box1_corners[1][0], box2_corners[1][0])
    inner_bottom_coord = min(box1_corners[1][1], box2_corners[1][1])

    # compute the area of intersection rectangle
    inter_area = abs(
        max((inner_right_coord - inner_left_coord, 0))
        * max((inner_bottom_coord - inner_top_coord), 0)
    )
    if inter_area == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    box1_area = abs(
        (box1_corners[0][0] - box1_corners[1][0]) * (box1_corners[0][1] - box1_corners[1][1])
    )
    box2_area = abs(
        (box2_corners[0][0] - box2_corners[1][0]) * (box2_corners[0][1] - box2_corners[1][1])
    )

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou


def overlay_object_2d_box_on_image(
    image, box2d: Box2D, color=(0, 0, 100), scale_factor=None, line_thickness=2
):
    """Visualize 2D box of annotated object on the image."""
    left_up = apply_scale(box2d.corners[0].astype(int), scale_factor)
    right_bottom = apply_scale(box2d.corners[2].astype(int), scale_factor)
    cv2.rectangle(image, left_up, right_bottom, color=color, thickness=line_thickness)
    return image


def overlay_object_3d_box_on_image(
    image,
    box3d: Box3D,
    calib,
    color=(0, 0, 100),
    scale_factor=None,
    line_thickness=2,
    camera: str = "camera_front",
):
    """Visualize 2D box of annotated object on the image."""
    box3d.convert_to(camera, calib)
    render_3d_box(image, box3d, calib, color, line_thickness)
    return image


def overlay_object_properties_on_image(
    image: np.ndarray,
    annotation: AnnotatedObject,
    color: Tuple[int, int, int] = (255, 0, 0),
    properties_list: List[str] = ["name"],
    scale_factor: float = 1,
    text_areas: list = [],
    text_scale: float = 0.9,
    object_id=-1,
) -> np.ndarray:
    """Visualize properties values for object."""

    properties = {
        "name": annotation.name,
        "object_type": annotation.object_type,
        "occlusion_level": annotation.occlusion_level,
        "object_id": object_id,
    }

    left_up = annotation.box2d.corners[0].astype(int)

    x_min, y_min = apply_scale(left_up, scale_factor)
    (width, height), _ = cv2.getTextSize("text", FONT_TYPE, text_scale, 2)
    # check text bounding box area
    text_area = [[x_min, y_min - height], [x_min, y_min]]
    text_lines = []
    for i, property_name in enumerate(properties_list):
        if property_name in properties:
            text = f"{property_name}:{properties[property_name]}"
        else:
            text = f"{property_name}:None"
        text_lines.append(text)
        (width, height), _ = cv2.getTextSize(text, FONT_TYPE, text_scale, 2)
        text_area[1][0] = max(text_area[1][0], x_min + width)
        text_area[1][1] = text_area[1][1] + height
    if text_areas and any(calc_iou(text_area, rec) for rec in text_areas):
        return image

    text_areas.append(text_area)

    # draw text
    for i, text in enumerate(text_lines):
        y = y_min + i * height
        if i == 0:
            (width, height), _ = cv2.getTextSize(text, FONT_TYPE, text_scale, 2)
            text_area = ((x_min, y - height), (x_min + width, y))
            cv2.rectangle(image, *text_area, color=(255, 255, 255), thickness=1)
        cv2.putText(image, text, (x_min, y), FONT_TYPE, text_scale, color, thickness=2)
    return image

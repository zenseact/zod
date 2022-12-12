"""This module contains functions for matching detections and ground truth.

As input we have a list of ground truth objects and a list of predictions.
The ground truth objects contains objects that correspond to dont-care regions,
for example parking lots, bike racks, etc. They can also be objects that does
not have any 3D properties (i.e., no 3D bounding box). This can be the case
for very distant objects that are visible in the camera but not in the LiDAR.
All these type of ground truth objects are ignored in the metric computation.

They are ignored by matching 3D false positives to these objects based on the
projection of the 3D bounding box into the image plane. These are matched with
ignore/dont-care objects by a IoU threshold.

The general outline of the matching process is as follows:
- Match the detections to the ground truth in 3D based on some metric.
- For all false positives in the 3D matching, we check the Birdeye View (BEV)
    intersection over detection-area (IoD) metric with all dont-care frustums
    (that also have been projected from camera 2D -> 3D -> BEV). Those false
    positives that have an IoD > over a certain threshold with a dont-care
    are removed from the false positives.
- Compute the relevant metrics based on the 3D matching and the
    false positives/false negatives.
"""

from dataclasses import dataclass
from shapely import geometry
from typing import Callable, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from zod.frames.evaluation.object_detection.utils import polygon_iod2D
from zod.constants import EVALUATION_FRAME
from zod.utils.objects import AnnotatedObject, PredictedObject
from zod.utils.zod_dataclasses import Calibration

GtPredMatch = Tuple[AnnotatedObject, PredictedObject]
MIN_CAMERA_ONLY_DEPTH = 150  # in meters
MAX_CAMERA_ONLY_DEPTH = 500  # in meters


@dataclass
class MatchedFrame:
    """Container for matching results."""

    matches: List[GtPredMatch]
    false_positives: List[PredictedObject]
    false_negatives: List[AnnotatedObject]


def split_gt_objects(
    ground_truth: List[AnnotatedObject],
) -> Tuple[List[AnnotatedObject], List[AnnotatedObject]]:
    """Split the ground truth objects into valid and dont-care objects."""

    # valid ground truth objects
    valid_gt: List[AnnotatedObject] = []
    # dont-care ground truth objects
    dont_care_gt: List[AnnotatedObject] = []
    for gt in ground_truth:
        if gt.should_ignore_object():
            dont_care_gt.append(gt)
        else:
            valid_gt.append(gt)

    return valid_gt, dont_care_gt


def match_dont_care_objects(
    dont_care_gt: List[AnnotatedObject],
    false_positives: List[PredictedObject],
    calibration: Calibration,
    fp_dist_fcn: Callable = polygon_iod2D,
    fp_dist_threshold: float = 0.25,
) -> List[PredictedObject]:
    """Match the false positives to the dont-care objects."""

    dont_care_polygons: List[geometry.Polygon] = []
    # turn all into frustums
    for gt in dont_care_gt:
        # if we have the 3d properties we can use the 3D box instead
        # of the frustum created from the 2D box.
        if gt.box3d is not None:
            bev_corners = gt.box3d.corners_bev
            bev_polygon = geometry.Polygon(bev_corners)
            dont_care_polygons.append(bev_polygon)
            continue
        # if we dont have 3d properties we use the 2D box
        # to create a frustum
        frustum = gt.box2d.get_3d_frustum(
            calibration,
            frame=EVALUATION_FRAME,
            min_depth=MIN_CAMERA_ONLY_DEPTH,
            max_depth=MAX_CAMERA_ONLY_DEPTH,
        )
        # ignore the z dimension as we move to bev
        frustum = frustum[:, :2]
        inner = frustum[:4]
        outer = frustum[4:]

        # initialize the polygon
        polygon_points = np.zeros((4, 2))
        # get the index of the minimum x value
        min_x_idx = np.argmin(inner[:, 0])
        # get the index of the maximum x value
        max_x_idx = np.argmax(inner[:, 0])
        # get the index of the minimum x value for the outer
        min_x_idx_outer = np.argmin(outer[:, 0])
        # get the index of the maximum x value for the outer
        max_x_idx_outer = np.argmax(outer[:, 0])
        polygon_points[0] = inner[min_x_idx]
        polygon_points[1] = inner[max_x_idx]
        polygon_points[2] = outer[max_x_idx_outer]
        polygon_points[3] = outer[min_x_idx_outer]

        dont_care_polygons.append(geometry.Polygon(polygon_points))

    match_to_dont_care: List[PredictedObject] = []

    for pred in false_positives:
        bev_corners = pred.box3d.corners_bev
        pred_poly = geometry.Polygon(bev_corners)

        for dc_polygon in dont_care_polygons:
            dist = fp_dist_fcn(pred_poly, dc_polygon)
            if dist > fp_dist_threshold:
                match_to_dont_care.append(pred)
                break

    return match_to_dont_care


def greedy_match(
    ground_truth: List[AnnotatedObject],
    predictions: List[PredictedObject],
    calibration: Calibration,
    dist_fcn: Callable,
    dist_threshold: float,
) -> MatchedFrame:
    """This function will greedily match the detections to the ground truth.

    It will use the confifence score to determine the order of the detections.

    Returns:
        - A list of matched ground truth objects and predictions.
        - A list of unmatched predictions.
        - A list of unmatched ground truth objects.
    """
    # trivial cases
    if len(ground_truth) == 0:
        return MatchedFrame([], predictions, [])
    if len(predictions) == 0:
        return MatchedFrame([], [], ground_truth)
    if len(ground_truth) == len(predictions) == 0:
        return MatchedFrame([], [], [])

    # Sort the predictions by their confidence score.
    predictions = sorted(predictions, key=lambda x: x.confidence, reverse=True)

    valid_gt, dont_care_gt = split_gt_objects(ground_truth)

    # Create a list of tuples. Each tuple contains the ground truth object and the prediction.
    matches: List[GtPredMatch] = []
    unmatched_predictions: List[PredictedObject] = []
    # Iterate over the predictions.
    for pred in predictions:
        # Iterate over all the ground truth objects and find the best match.
        min_dist = np.inf
        best_match_idx = None
        for gt_idx, gt in enumerate(valid_gt):
            distance = dist_fcn(gt.box3d, pred.box3d)
            if distance < min_dist:
                min_dist = distance
                best_match_idx = gt_idx

        if min_dist > dist_threshold:
            unmatched_predictions.append(pred)
        else:
            # We have a match.
            matches.append((valid_gt.pop(best_match_idx), pred))

    if len(unmatched_predictions) > 0:
        dont_care_matches = match_dont_care_objects(
            dont_care_gt, unmatched_predictions, calibration
        )

        # Remove the matched false positives from the unmatched predictions.
        unmatched_predictions = [
            pred for pred in unmatched_predictions if (pred not in dont_care_matches)
        ]

    # return the matches, the unmatched predictions and the remaining valid ground truth objects
    return MatchedFrame(
        matches=matches,
        false_positives=unmatched_predictions,
        false_negatives=valid_gt,
    )


def optimal_match(
    ground_truth: List[AnnotatedObject],
    predictions: List[PredictedObject],
    calibration: Calibration,
    dist_fcn: Callable,
    dist_threshold: float,
) -> MatchedFrame:
    """This function will make an optimal matching between the detections and the ground truth.

    Use the Hungarian algorithm to find the optimal matching.
    """

    if len(ground_truth) == 0:
        return MatchedFrame([], predictions, [])
    if len(predictions) == 0:
        return MatchedFrame([], [], ground_truth)
    if len(ground_truth) == len(predictions) == 0:
        return MatchedFrame([], [], [])

    if len(ground_truth):
        # assert that all ground truth objects are in the same frame
        assert len(set([gt.box3d.frame for gt in ground_truth])) == 1
    if len(predictions):
        # assert that all predictions are in the same frame
        assert len(set([pred.box3d.frame for pred in predictions])) == 1
    if len(ground_truth) and len(predictions):
        # assert that the frames are the same
        assert ground_truth[0].box3d.frame == predictions[0].box3d.frame

    valid_gt, dont_care_gt = split_gt_objects(ground_truth)

    # initialize the cost matrix
    cost_matrix = np.zeros((len(valid_gt), len(predictions)))
    for gt_idx, gt in enumerate(ground_truth):
        for pred_idx, pred in enumerate(predictions):
            dist = dist_fcn(gt.box3d, pred.box3d)
            if dist > dist_threshold:
                cost_matrix[gt_idx, pred_idx] = np.inf
            else:
                cost_matrix[gt_idx, pred_idx] = -(dist + np.log(pred.confidence))

    # find the optimal matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # create the matches
    matches: List[GtPredMatch] = []
    unmatched_predictions: List[PredictedObject] = []
    for gt_idx, pred_idx in zip(row_ind, col_ind):
        if cost_matrix[gt_idx, pred_idx] == np.inf:
            unmatched_predictions.append(predictions[pred_idx])
        else:
            matches.append((valid_gt[gt_idx], predictions[pred_idx]))

    match_to_dont_care = match_dont_care_objects(dont_care_gt, unmatched_predictions, calibration)

    # Remove the matched false positives from the unmatched predictions.
    unmatched_predictions = [
        pred for pred in unmatched_predictions if pred not in match_to_dont_care
    ]

    # return the matches, the unmatched predictions and the remaining valid ground truth objects
    return MatchedFrame(
        matches=matches,
        false_positives=unmatched_predictions,
        false_negatives=valid_gt,
    )


def match_one_frame(
    ground_truth: List[AnnotatedObject],
    predictions: List[PredictedObject],
    calibration: Calibration,
    dist_fcn: Callable,
    dist_threshold: float,
    method: str = "greedy",
) -> MatchedFrame:
    """Match the detections to the ground truth.

    Args:
        ground_truth: A list of ground truth objects.
        predictions: A list of predictions.
        calibration: The calibration of the scene.
        dist_fcn: The distance function to use. should take 2 Box3D objects as arguments
        dist_threshold: The distance threshold.
        method: The method to use. Can be "greedy" or "optimal".
    Returns:
        A list of tuples. Each tuple contains the ground truth object and the prediction.
    """
    # check that all objects are in the evaluation frame
    assert all(
        gt.box3d.frame == EVALUATION_FRAME for gt in ground_truth if gt.box3d is not None
    ), "All ground truth objects must be in the evaluation frame."
    assert all(
        pred.box3d.frame == EVALUATION_FRAME for pred in predictions
    ), "All predictions must be in the evaluation frame."

    if method == "greedy":
        return greedy_match(
            ground_truth=ground_truth,
            predictions=predictions,
            calibration=calibration,
            dist_fcn=dist_fcn,
            dist_threshold=dist_threshold,
        )
    elif method == "optimal":
        return optimal_match(
            ground_truth=ground_truth,
            predictions=predictions,
            calibration=calibration,
            dist_fcn=dist_fcn,
            dist_threshold=dist_threshold,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

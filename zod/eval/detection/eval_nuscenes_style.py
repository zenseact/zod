"""Functions for evaluating object detection models."""

import json
import os
from typing import Callable, Dict

import numpy as np

from zod.eval.detection._nuscenes_eval.common.data_classes import EvalBoxes
from zod.eval.detection._nuscenes_eval.common.utils import center_distance
from zod.eval.detection._nuscenes_eval.detection.algo import accumulate, calc_ap, calc_tp
from zod.eval.detection._nuscenes_eval.detection.data_classes import (
    DetectionBox,
    DetectionConfig,
    DetectionMetrics,
)
from zod.eval.detection.constants import EVALUATION_CLASSES

VALID_TP_METRICS = {"trans_err": "mATE", "scale_err": "mASE", "orient_err": "mAOE"}
PRECISION_RECALL_SAMPLING_POINTS = 101

NUSCENES_SUMMARY = """
NUSCENES METRIC SUMMARY:
distance function: {dist_func}
distance thresholds: {dist_thresh}
--------------------------------------------------
NDS:  {NDS:.4f}
mAP:  {mAP:.4f}
mATE: {mATE:.4f}
mASE: {mASE:.4f}
mAOE: {mAOE:.4f}"""

PER_CLASS_SUMMARY = """
{cls}:
    mAP:  {mAP:.4f}
    mATE: {mATE:.4f}
    mASE: {mASE:.4f}
    mAOE: {mAOE:.4f}"""

# for reference
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

# our settings
ZOD_DEFAULT_SETTINGS = {
    "class_range": {cls_name: 250 for cls_name in EVALUATION_CLASSES},
    "dist_fcn": "center_distance",
    "dist_ths": [0.5, 1.0, 2.0, 4.0],
    "dist_th_tp": 2.0,
    "min_recall": 0.1,
    "min_precision": 0.1,
    "max_boxes_per_sample": 500,
    "mean_ap_weight": 5,
}


def evaluate_nuscenes_style(
    gt_boxes: EvalBoxes,
    det_boxes: EvalBoxes,
    verbose: bool = False,
    output_path: str = None,
    verify_coordinate_system: bool = True,
) -> Dict[str, float]:
    """Perform nuscenes evaluation based on a number of groundtruths and detections."""
    if verify_coordinate_system:
        _check_coordinate_system(gt_boxes)

    detection_cfg = DetectionConfig(**ZOD_DEFAULT_SETTINGS)
    detection_metrics = DetectionMetrics(detection_cfg)

    class_ranges = {k: (0, v) for k, v in ZOD_DEFAULT_SETTINGS["class_range"].items()}

    # filter according to the default nuscenes settings
    gt_boxes = _filter_eval_boxes_on_ranges(gt_boxes, class_ranges)
    det_boxes = _filter_eval_boxes_on_ranges(det_boxes, class_ranges)

    metrics = {dist_th: _nuscenes_evaluate(gt_boxes, det_boxes, dist_th=dist_th) for dist_th in detection_cfg.dist_ths}

    evaluated_clses = set(metrics[detection_cfg.dist_ths[0]].keys())
    for zod_cls in evaluated_clses:
        # They evaluate the ap across all thresholds
        for dist_th in detection_cfg.dist_ths:
            detection_metrics.add_label_ap(
                detection_name=zod_cls,
                dist_th=dist_th,
                ap=metrics[dist_th][zod_cls]["ap"],
            )
        # They evaluate the tp across only one threshold
        for metric in VALID_TP_METRICS:
            detection_metrics.add_label_tp(zod_cls, metric, metrics[detection_cfg.dist_th_tp][zod_cls][metric])

    if verbose:
        _print_nuscenes_metrics(detection_metrics)

    serialized = _serialize(detection_metrics)

    if output_path:
        with open(os.path.join(output_path, "nuscenes_evaluation_metrics.json"), "w") as f:
            json.dump(serialized, f)

    return serialized


def _check_coordinate_system(gt_boxes: EvalBoxes):
    """Use heuristics to check if the boxes were provided in the ego coordinate system."""
    # The key assumption is x axis has the longest range in the ego coordinate system
    # whereas for camera and lidar the z and y axis are the longest range respectively
    centers = np.array([box.translation for box in gt_boxes.all])
    center_means = np.mean(centers, axis=0)
    max_avgdist_axis = np.argmax(center_means)
    error_msg = (
        "Looks like the boxes were provided in the {sensor} coordinate system. "
        "Please convert them to the ego coordinate system. If you know what you are doing, "
        " you can disable this check by setting `verify_coordinate_system=False`."
    )
    desired_axis = 0  # EGO
    if max_avgdist_axis == 0 and desired_axis != 0:
        raise ValueError(error_msg.format(sensor="ego"))
    elif max_avgdist_axis == 1 and desired_axis != 1:
        raise ValueError(error_msg.format(sensor="lidar"))
    elif max_avgdist_axis == 2 and desired_axis != 2:
        raise ValueError(error_msg.format(sensor="camera"))


def _nuscenes_evaluate(
    gt_boxes: EvalBoxes,
    det_boxes: EvalBoxes,
    dist_fcn: Callable = center_distance,
    dist_th: float = 1,
    min_precision: float = 0.1,
    min_recall: float = 0.1,
) -> Dict[str, float]:
    """Perform nuscenes evaluation based on a number of groundtruths and detections."""
    metrics = {}

    existing_classes = set()
    for box in gt_boxes.all:
        existing_classes.add(box.detection_name)

    for cls in EVALUATION_CLASSES:
        if cls not in existing_classes:
            # ensure that we have samples for this class in the gt data
            continue

        md = accumulate(
            gt_boxes,
            det_boxes,
            cls,
            dist_fcn=dist_fcn,
            dist_th=dist_th,
        )

        metrics[cls] = {metric: calc_tp(md, min_recall=0.1, metric_name=metric) for metric in VALID_TP_METRICS}
        metrics[cls]["ap"] = calc_ap(md, min_recall=min_recall, min_precision=min_precision)
    return metrics


def _filter_eval_boxes_on_ranges(boxes: EvalBoxes, class_ranges: Dict[str, int], verbose: bool = False) -> EvalBoxes:
    """Filter out boxes that are outside of the range of the classes."""
    filtered_boxes = EvalBoxes()

    def _filter(box: DetectionBox):
        return (
            box.detection_name in class_ranges
            and class_ranges[box.detection_name][0] < box.ego_dist <= class_ranges[box.detection_name][1]
        )

    for frame_id in boxes.sample_tokens:
        filtered_boxes.add_boxes(frame_id, [box for box in boxes[frame_id] if _filter(box)])
    if verbose:
        # print the number of removed boxes per class
        for cls in class_ranges:
            n_originally = sum(1 for box in boxes.all if box.detection_name == cls)
            n_filtered = sum(1 for box in filtered_boxes.all if box.detection_name == cls)
            n_removed = n_originally - n_filtered
            if n_removed > 0:
                print(f"{cls}: from {n_originally} -> {n_filtered} ({n_removed} removed)")

    return filtered_boxes


def _print_nuscenes_metrics(metrics: DetectionMetrics):
    print(
        NUSCENES_SUMMARY.format(
            dist_func=metrics.cfg.dist_fcn,
            dist_thresh=metrics.cfg.dist_ths,
            NDS=metrics.nd_score,
            mAP=metrics.mean_ap,
            mATE=metrics.tp_errors["trans_err"],
            mASE=metrics.tp_errors["scale_err"],
            mAOE=metrics.tp_errors["orient_err"],
        )
    )
    for cls in metrics.mean_dist_aps:
        print(
            PER_CLASS_SUMMARY.format(
                cls=cls,
                mAP=metrics.mean_dist_aps[cls],
                mATE=metrics._label_tp_errors[cls]["trans_err"],
                mASE=metrics._label_tp_errors[cls]["scale_err"],
                mAOE=metrics._label_tp_errors[cls]["orient_err"],
            )
        )


def _serialize(detection_metrics: DetectionMetrics) -> Dict[str, float]:
    # Only serialize the classes that were evaluated (had GT)
    classes = list(detection_metrics.mean_dist_aps.keys())
    tp_metrics = {name: detection_metrics.tp_errors[metric] for metric, name in VALID_TP_METRICS.items()}
    class_aps = {f"{cls}/mAP": detection_metrics.mean_dist_aps[cls] for cls in classes}
    class_tps = {
        f"{cls}/{name}": detection_metrics._label_tp_errors[cls][metric]
        for cls in classes
        for metric, name in VALID_TP_METRICS.items()
    }
    return {
        "NDS": detection_metrics.nd_score,
        "mAP": detection_metrics.mean_ap,
        **tp_metrics,
        **class_aps,
        **class_tps,
    }

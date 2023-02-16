"""Functions for evaluating object detection models."""

import json
import os
from typing import Callable, Dict

from zod.eval.detection._nuscenes_eval.common.data_classes import EvalBoxes
from zod.eval.detection._nuscenes_eval.common.utils import center_distance
from zod.eval.detection._nuscenes_eval.detection.algo import accumulate, calc_ap, calc_tp
from zod.eval.detection._nuscenes_eval.detection.data_classes import (
    DetectionBox,
    DetectionConfig,
    DetectionMetrics,
)
from zod.eval.detection.constants import EVALUATION_CLASSES

VALID_TP_METRICS = ["trans_err", "scale_err", "orient_err"]
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

    for cls in EVALUATION_CLASSES:
        # ensure that we have samples for this class in the gt data
        n_samples = sum(1 for box in gt_boxes.all if box.detection_name == cls)
        if n_samples == 0:
            continue
        md = accumulate(
            gt_boxes,
            det_boxes,
            cls,
            dist_fcn=dist_fcn,
            dist_th=dist_th,
        )

        metrics[cls] = {
            metric: calc_tp(md, min_recall=0.1, metric_name=metric) for metric in VALID_TP_METRICS
        }
        metrics[cls]["ap"] = calc_ap(md, min_recall=min_recall, min_precision=min_precision)
    return metrics


def _filter_eval_boxes_on_ranges(
    boxes: EvalBoxes, class_ranges: Dict[str, int], verbose: bool = False
) -> EvalBoxes:
    """Filter out boxes that are outside of the range of the classes."""
    filtered_boxes = EvalBoxes()

    def _filter(box: DetectionBox):
        return (
            box.detection_name in class_ranges
            and class_ranges[box.detection_name][0]
            < box.ego_dist
            <= class_ranges[box.detection_name][1]
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


def print_nuscenes_metrics(metrics: DetectionMetrics):
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


def evaluate_nuscenes_style(
    gt_boxes: EvalBoxes,
    det_boxes: EvalBoxes,
    verbose: bool = False,
    output_path: str = None,
) -> DetectionMetrics:
    """Perform nuscenes evaluation based on a number of groundtruths and detections."""
    detection_cfg = DetectionConfig(**NUSCENES_DEFAULT_SETTINGS)
    detection_metrics = DetectionMetrics(detection_cfg)

    class_ranges = {k: (0, v) for k, v in NUSCENES_DEFAULT_SETTINGS["class_range"].items()}

    # filter according to the default nuscenes settings
    gt_boxes = _filter_eval_boxes_on_ranges(gt_boxes, class_ranges)
    det_boxes = _filter_eval_boxes_on_ranges(det_boxes, class_ranges)

    metrics = {
        dist_th: _nuscenes_evaluate(gt_boxes, det_boxes, dist_th=dist_th)
        for dist_th in detection_cfg.dist_ths
    }

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
            detection_metrics.add_label_tp(
                zod_cls, metric, metrics[detection_cfg.dist_th_tp][zod_cls][metric]
            )

    if output_path:
        with open(os.path.join(output_path, "nuscenes_evaluation_metrics.json"), "w") as f:
            json.dump(detection_metrics.serialize(), f)

    if verbose:
        print_nuscenes_metrics(detection_metrics)

    return detection_metrics

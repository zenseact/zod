"""Functions for evaluating object detection models."""

import json
import os
from itertools import repeat
from typing import Callable, Dict, Iterator, List, Tuple

import numpy as np
from tqdm.contrib.concurrent import process_map

from zod.frames.evaluation.object_detection.matching import MatchedFrame, match_one_frame
from zod.frames.evaluation.object_detection.nuscenes_eval.common.data_classes import EvalBoxes
from zod.frames.evaluation.object_detection.nuscenes_eval.common.utils import center_distance
from zod.frames.evaluation.object_detection.nuscenes_eval.detection.algo import (
    accumulate,
    calc_ap,
    calc_tp,
)
from zod.frames.evaluation.object_detection.nuscenes_eval.detection.data_classes import (
    DetectionBox,
    DetectionConfig,
    DetectionMetrics,
)
from zod.frames.evaluation.object_detection.utils import NUSCENES_DEFAULT_SETTINGS
from zod.constants import EVALUATION_CLASSES
from zod.utils.objects import AnnotatedObject, PredictedObject
from zod.utils.zod_dataclasses import Calibration


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

EvalFrame = Tuple[List[AnnotatedObject], List[PredictedObject], Calibration]


class Evalset:
    def __init__(self):
        self._frames: Dict[str, EvalFrame] = {}

    def add_frame(self, frame_id: str, frame: EvalFrame):
        self._frames[frame_id] = frame

    def __len__(self):
        return len(self._frames)

    def __repr__(self) -> str:
        return f"Evalset with {len(self)} frames."

    def __iter__(self) -> Iterator[Tuple[str, EvalFrame]]:
        return iter(self._frames.items())

    def __getitem__(self, frame_id: str) -> EvalFrame:
        return self._frames[frame_id]

    @property
    def frames(self) -> Dict[str, EvalFrame]:
        """Return all frames."""
        return self._frames


def nuscenes_evaluate(
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


def filter_eval_boxes_on_ranges(
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


def original_nuscenes_eval(
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
    gt_boxes = filter_eval_boxes_on_ranges(gt_boxes, class_ranges)
    det_boxes = filter_eval_boxes_on_ranges(det_boxes, class_ranges)

    metrics = {
        dist_th: nuscenes_evaluate(gt_boxes, det_boxes, dist_th=dist_th)
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


def _evaluate_frame(
    frame_id: str,
    frame: EvalFrame,
    cls: str,
    dist_fcn: Callable,
    dist_th: float,
) -> Dict[str, MatchedFrame]:

    gts, preds, calibration = frame
    # filter out all predictions that are not of the correct class
    preds = [pred for pred in preds if pred.name == cls]
    # filter the ground truth boxes based on cls
    gts = [gt for gt in gts if gt.name == cls]

    matched_frame = match_one_frame(
        ground_truth=gts,
        predictions=preds,
        calibration=calibration,
        dist_fcn=dist_fcn,
        dist_threshold=dist_th,
    )

    return {frame_id: matched_frame}


def zod_evaluation(
    evalset: Evalset,
    dist_fcn: Callable,
    dist_th: float,
):
    # evaluate over all classes
    for cls in EVALUATION_CLASSES:
        # process all frames in parallel
        results: List[Dict[str, MatchedFrame]] = process_map(
            _evaluate_frame,
            evalset.frames.keys(),
            evalset.frames.values(),
            repeat(cls),
            repeat(dist_fcn),
            repeat(dist_th),
            desc=f"Accumulating metrics for class: {cls}",
            chunksize=1 if len(evalset.frames) < 100 else 100,
        )

        # join the list of dicts into a single dict
        results: Dict[str, MatchedFrame] = {k: v for d in results for k, v in d.items()}

        # accumulate the metrics across all frames
        tps = []
        fps = []
        confs = []
        n_ground_truths = 0

        for _, matched_frame in results.items():
            n_ground_truths += len(matched_frame.matches) + len(matched_frame.false_negatives)
            for _, pred in matched_frame.matches:
                tps.append(1)
                fps.append(0)
                confs.append(pred.confidence)
            for pred in matched_frame.false_positives:
                tps.append(0)
                fps.append(1)
                confs.append(pred.confidence)

        # sort the accumulated list based on confidence
        confs, tps, fps = zip(*sorted(zip(confs, tps, fps), reverse=True))
        # accumulate the tps and fps
        tps = np.cumsum(tps).astype(np.float)
        fps = np.cumsum(fps).astype(np.float)
        # compute the precision and recall
        precision = tps / (tps + fps)
        recall = tps / n_ground_truths

        raise NotImplementedError("AP computation is not yet implemented")
        # compute the ap
        recall_sampling_points = np.linspace(0, 1, PRECISION_RECALL_SAMPLING_POINTS)
        precision = np.interp(recall_sampling_points, recall, precision, right=0)
        # confidences = np.interp(recall_sampling_points, recall, confs, right=0)
        recall = recall_sampling_points

        # store in some sort of data structure, both the raw data but also the computed metrics
        # todo (william), store the results here
        ap = np.mean(precision)
        print(f"AP for class {cls} is {ap}")

    # compute the final metrics, average over classes etc
    # todo (william), compute the final metrics here

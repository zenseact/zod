# raise NotImplementedError("This evaluation framework is not yet implemented")

from itertools import repeat
from typing import Callable, Dict, Iterator, List, Tuple

import numpy as np
from tqdm.contrib.concurrent import process_map

from zod.anno.object import ObjectAnnotation
from zod.data_classes.calibration import Calibration
from zod.eval.detection._experimental.matching import MatchedFrame
from zod.eval.detection._experimental.utils import PredictedObject
from zod.eval.detection.constants import EVALUATION_CLASSES

EvalFrame = Tuple[List[ObjectAnnotation], List[PredictedObject], Calibration]


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


def _evaluate_frame(
    frame_id: str,
    frame: EvalFrame,
    cls: str,
    dist_fcn: Callable,
    dist_th: float,
) -> Dict:  # Dict[str, MatchedFrame]:
    from zod.eval.detection.matching import MatchedFrame, match_one_frame

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
        eval_classes=EVALUATION_CLASSES,
    )

    return {frame_id: matched_frame}


def evaluate(
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

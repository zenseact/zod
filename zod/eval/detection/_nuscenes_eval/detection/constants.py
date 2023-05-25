# nuScenes dev-kit.
# Code written by Oscar Beijbom and Varun Bankiti, 2019.

from zod.eval.detection.constants import EVALUATION_CLASSES


DETECTION_NAMES = EVALUATION_CLASSES

TP_METRICS = ["trans_err", "scale_err", "orient_err"]

PRETTY_TP_METRICS = {
    "trans_err": "Trans.",
    "scale_err": "Scale",
    "orient_err": "Orient.",
}

TP_METRICS_UNITS = {
    "trans_err": "m",
    "scale_err": "1-IOU",
    "orient_err": "rad.",
}

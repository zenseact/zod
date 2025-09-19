"""Calls the save ground truth function with specified configs."""

from dataset.groundtruth_utils import save_ground_truth
from dataset.zod_configs import ZodConfigs

from zod import constants
from zod.zod_frames import ZodFrames


def generate_ground_truth() -> None:
    """Create ground truth."""
    zod_configs = ZodConfigs()
    zod_frames = ZodFrames(dataset_root=zod_configs.DATASET_ROOT, version="full")
    training_frames_all = zod_frames.get_split(constants.TRAIN)
    validation_frames_all = zod_frames.get_split(constants.VAL)

    save_ground_truth(zod_frames, training_frames_all, validation_frames_all, zod_configs=zod_configs)

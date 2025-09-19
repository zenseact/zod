"""Groundtruth utilities."""

import json
from typing import Set

import numpy as np
from dataset.zod_configs import ZodConfigs
from tqdm import tqdm

from zod import ZodFrames


def get_ground_truth(zod_frames: ZodFrames, frame_id: int, zod_configs: ZodConfigs) -> np.array:
    """Get true holistic path from future GPS locations.

    Args:
        zod_frames (ZodDataset): ZodDataset
        frame_id (int): frame id
        zod_configs (ZodConfigs): zod configs dataclass

    Returns:
        np.array: true path

    """
    zod_frame = zod_frames[frame_id]
    oxts = zod_frame.oxts
    key_timestamp = zod_frame.info.keyframe_time.timestamp()

    # get posses associated with frame timestamp
    try:
        current_pose = oxts.get_poses(key_timestamp)
        all_poses = oxts.poses[oxts.timestamps >= key_timestamp]
        transformed_poses = np.linalg.pinv(current_pose) @ all_poses
        translations = transformed_poses[:, :3, 3]
        distances = np.linalg.norm(np.diff(translations, axis=0), axis=1)
        accumulated_distances = np.cumsum(distances).astype(int).tolist()

        # get the poses that each have a point having a distance from TARGET_DISTANCES
        pose_idx = [accumulated_distances.index(i) for i in zod_configs.TARGET_DISTANCES]
        used_poses = transformed_poses[pose_idx]

    except Exception as _:
        print(f"detected invalid frame: {frame_id}")
        return np.array([])

    points = used_poses[:, :3, -1]
    return points.flatten()


def save_ground_truth(
    zod_frames: ZodFrames,
    training_frames: Set[str],
    validation_frames: Set[str],
    zod_configs: ZodConfigs,
) -> None:
    """Write ground truth as json.

    Args:
        zod_frames (ZodDataset): _description_
        training_frames (Set[str]): _description_
        validation_frames (Set[str]): _description_
        zod_configs (ZodConfigs): zod configs dataclass

    """
    all_frames = validation_frames.copy()
    all_frames.update(training_frames)

    corrupted_frames = []
    ground_truths = {}
    for frame_id in tqdm(all_frames):
        ground_truth = get_ground_truth(zod_frames, frame_id, zod_configs)

        if ground_truth.shape[0] != zod_configs.NUM_OUTPUT:
            corrupted_frames.append(frame_id)
            continue

        ground_truths[frame_id] = ground_truth.tolist()

    # Serializing json
    json_object = json.dumps(ground_truths, indent=4)

    # Writing to sample.json
    with open(zod_configs.STORED_GROUND_TRUTH_PATH, "w") as outfile:
        outfile.write(json_object)

    print(f"{corrupted_frames}")


def load_ground_truth(path: str) -> dict:
    """Load ground truth from file."""
    with open(path) as json_file:
        gt = json.load(json_file)

    for f in gt:
        gt[f] = np.array(gt[f])

    return gt

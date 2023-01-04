"""Compensate lidar point clouds."""

import json
import os
import os.path as osp
from functools import partial
from typing import Tuple

import typer
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from zod.utils.compensation import motion_compensate_pointwise, motion_compensate_scanwise
from zod.dataclasses.oxts import EgoMotion
from zod.utils.utils import parse_datetime_from_filename
from zod.dataclasses.zod_dataclasses import Calibration, LidarData


OLD_LIDAR_FOLDER = "lidar_velodyne"
COMPENSATED_LIDAR_FOLDER = "lidar_velodyne_compensated"
GLOBAL_LIDAR_FOLDER = "lidar_velodyne_global"
KEYFRAME_LIDAR_FOLDER = "lidar_velodyne_keyframe"
OXTS_FOLDER = "oxts"
IMAGE_FOLDER = "camera_front_original"
CALIBRATION_FOLDER = "calibration"


def main(
    sequences_dir: str = typer.Argument(..., help="Path to the lidar data"),
    save_global: bool = typer.Option(False, help="Whether to save global point clouds"),
    compensate_to_keyframe: bool = typer.Option(True, help="Whether to compensate to keyframe"),
    partitions: int = typer.Option(1, help="Number of partitions to split the data into"),
    current_partition: int = typer.Option(0, help="Current partition to process"),
    clouds_in_global: int = typer.Option(
        -1, help="Number of clouds around keyframe to use for global compensation"
    ),
):
    sequences = sorted(list(os.listdir(sequences_dir)))
    for sequence in tqdm(sequences[current_partition::partitions], desc="Processing sequences..."):
        sequence_dir = osp.join(sequences_dir, sequence)
        oxts_file = os.listdir(osp.join(sequence_dir, OXTS_FOLDER))
        if not oxts_file:
            print("Skipping due to missing oxts:", sequence)
            continue
        calibration_file = os.listdir(osp.join(sequence_dir, CALIBRATION_FOLDER))
        if not calibration_file:
            print("Skipping due to missing calibration:", sequence)
            continue
        lidar_files = os.listdir(osp.join(sequence_dir, OLD_LIDAR_FOLDER))
        if not len(lidar_files) > 180:
            print("Skipping due to too few lidar files:", sequence)
            continue
        # TODO: maybe wipe this folder instead of continuing above
        # Extract ego motion
        oxts_file = sorted(oxts_file)[-1]
        ego_motion = EgoMotion.from_sequence_oxts(osp.join(sequence_dir, OXTS_FOLDER, oxts_file))

        # Extract calibration
        calibration_file = calibration_file[0]
        with open(osp.join(sequence_dir, CALIBRATION_FOLDER, calibration_file), "r") as f:
            calibration = Calibration.from_dict(json.load(f))
        calibration = calibration.lidars["lidar_velodyne"]

        if compensate_to_keyframe and not _get_num_files(
            osp.join(sequence_dir, KEYFRAME_LIDAR_FOLDER)
        ):
            compensate_to_keyframe_cloud(
                sequence, sequence_dir, lidar_files, ego_motion, calibration
            )
        if _get_num_files(osp.join(sequence_dir, COMPENSATED_LIDAR_FOLDER)) != len(lidar_files):
            comp_func = partial(
                compensate_single_cloud,
                sequence_dir=sequence_dir,
                ego_motion=ego_motion,
                calibration=calibration,
            )
            process_map(comp_func, lidar_files, chunksize=1, desc=f"Compensating {sequence}...")
        if save_global:
            save_global_cloud(sequence, sequence_dir, ego_motion, calibration, clouds_in_global)


def compensate_single_cloud(lidar_file, sequence_dir, ego_motion, calibration):
    """Extract and compensate a single lidar point."""
    lidar_timestamp, lidar = _read_lidar(osp.join(sequence_dir, OLD_LIDAR_FOLDER), lidar_file)
    new_lidar = motion_compensate_pointwise(lidar, ego_motion, calibration, lidar_timestamp)
    new_lidar_path = osp.join(sequence_dir, COMPENSATED_LIDAR_FOLDER, lidar_file)
    os.makedirs(osp.dirname(new_lidar_path), exist_ok=True)
    new_lidar.to_npy(new_lidar_path)


def save_global_cloud(sequence, sequence_dir, ego_motion, calibration, clouds_in_global):
    """Aggregate all lidar point clouds into one global cloud."""
    global_lidar = LidarData.empty()
    # Extract and compensate lidar point clouds
    lidar_dir = osp.join(sequence_dir, COMPENSATED_LIDAR_FOLDER)
    lidar_files = sorted(os.listdir(lidar_dir))
    if clouds_in_global != -1:
        lidar_files = lidar_files[
            len(lidar_files) // 2 - clouds_in_global : len(lidar_files) // 2 + clouds_in_global
        ]
    global_ts = parse_datetime_from_filename(lidar_files[len(lidar_files) // 2]).timestamp()
    for lidar_file in tqdm(
        lidar_files, desc=f"Storing global cloud for {sequence}...", leave=False
    ):
        _, lidar = _read_lidar(lidar_dir, lidar_file)
        global_lidar.append(motion_compensate_scanwise(lidar, ego_motion, calibration, global_ts))
    # Write each time which is not ideal but it's a one-time script
    global_lidar_path = osp.join(sequence_dir, GLOBAL_LIDAR_FOLDER, f"{global_ts}.npy")
    os.makedirs(osp.dirname(global_lidar_path), exist_ok=True)
    global_lidar.to_npy(global_lidar_path)


def compensate_to_keyframe_cloud(sequence, sequence_dir, lidar_files, ego_motion, calibration):
    """Extract and compensate the nearest point cloud to the keyframe (middle) image."""
    print("Compensating to keyframe for: ", sequence)
    images = os.listdir(osp.join(sequence_dir, IMAGE_FOLDER))
    images = [i for i in images if i.endswith(".jpg")]
    keyframe_image_name = images[len(images) // 2]
    keyframe_datetime = parse_datetime_from_filename(keyframe_image_name)
    nearest_lidar = min(
        lidar_files, key=lambda x: abs(parse_datetime_from_filename(x) - keyframe_datetime)
    )
    _, lidar = _read_lidar(osp.join(sequence_dir, OLD_LIDAR_FOLDER), nearest_lidar)
    pbar = tqdm(total=1, desc=f"Compensating to keyframe for {sequence}...", leave=False)
    new_lidar = motion_compensate_pointwise(
        lidar, ego_motion, calibration, keyframe_datetime.timestamp()
    )
    new_lidar_path = osp.join(
        sequence_dir, KEYFRAME_LIDAR_FOLDER, keyframe_image_name.replace(".jpg", ".npy")
    )
    os.makedirs(osp.dirname(new_lidar_path), exist_ok=True)
    new_lidar.to_npy(new_lidar_path)
    pbar.update(1)
    pbar.close()


def _read_lidar(lidar_dir, lidar_file) -> Tuple[float, LidarData]:
    lidar_path = osp.join(lidar_dir, lidar_file)
    lidar_datetime = parse_datetime_from_filename(lidar_path)
    lidar_timestamp = lidar_datetime.timestamp()
    lidar = LidarData.from_npy(lidar_path)
    return lidar_timestamp, lidar


def _get_num_files(folder: str) -> int:
    """Return number of files if folder exists, otherwise 0."""
    if os.path.exists(folder):
        return len(os.listdir(folder))
    return 0


if __name__ == "__main__":
    typer.run(main)

import json
import os
import os.path as osp
import shutil
from itertools import repeat

import h5py
import numpy as np
import typer
from tqdm.contrib.concurrent import process_map

from zod.zod_dataclasses.frame import ZodFrame
from zod.zod_dataclasses.info import Information
from zod.zod_dataclasses.oxts import EgoMotion
from zod.zod_dataclasses.sequence import ZodSequence
from zod.zod_frames import ZodFrames
from zod.zod_sequences import ZodSequences

DATASET_ROOT = "/staging/dataset_donation/round_2/"
FILE_NAME = "ego_motion.json"
SEQUENCES = "sequences"
FRAMES = "single_frames"


def interpolate_and_write_oxts(info: Information, oxts: EgoMotion, save_dir: str):
    datetimes = [frame.time for frame in info.all_frames]
    datetimes = sorted(list(set(datetimes)))
    timestamps = np.array([t.timestamp() for t in datetimes])

    ego_motion_interp = oxts.interpolate(timestamps)
    with open(osp.join(save_dir, info.id, "ego_motion.json"), "w") as f:
        json.dump(ego_motion_interp.to_json(), f)


def process_frame(frame: ZodFrame, dataset_root: str, write_poses_to_oxts: bool):
    oxts = frame.oxts
    try:
        interpolate_and_write_oxts(frame.info, oxts, save_dir=osp.join(dataset_root, FRAMES))
    except Exception as e:
        print(f"Error in {frame.info.id}: {e}")
    if write_poses_to_oxts:
        with h5py.File(frame.info.oxts_path, "r") as f:
            if "poses" in f:
                print("poses already in oxts file")
                return
        backup_path = frame.info.oxts_path.replace("/oxts/", "/oxts_old/")
        if not osp.exists(backup_path):
            os.makedirs(osp.dirname(backup_path), exist_ok=True)
            shutil.copyfile(frame.info.oxts_path, backup_path)
        with h5py.File(frame.info.oxts_path, "a") as f:
            f.create_dataset("poses", data=oxts.poses)


def process_sequence(sequence: ZodSequence, dataset_root: str):
    oxts = sequence.oxts
    interpolate_and_write_oxts(sequence.info, oxts, save_dir=osp.join(dataset_root, SEQUENCES))


def main(
    dataset_root: str = typer.Option(DATASET_ROOT),
    sequences: bool = typer.Option(False),
    frames: bool = typer.Option(False),
    write_poses_to_oxts: bool = typer.Option(False),
):
    if not sequences and not frames:
        raise ValueError("Please specify either --sequences or --frames")
    if sequences:
        zod_sequences = ZodSequences(dataset_root=dataset_root, version="full")
        process_map(
            process_sequence,
            zod_sequences,
            repeat(dataset_root),
            chunksize=100,
            desc="Processing sequences",
        )
        # for seq in zod_sequences:
        #     process_sequence(seq, dataset_root)
    if frames:
        zod_frames = ZodFrames(dataset_root=dataset_root, version="full")
        process_map(
            process_frame,
            zod_frames,
            repeat(dataset_root),
            repeat(write_poses_to_oxts),
            chunksize=100,
            desc="Processing frames",
        )
        # for frame in zod_frames:
        #     process_frame(frame, dataset_root, write_poses_to_oxts)


if __name__ == "__main__":
    typer.run(main)

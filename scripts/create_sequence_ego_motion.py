import json
import os.path as osp

import numpy as np

from zod.constants import Lidar
from zod.dataclasses.frame import ZodFrame
from zod.dataclasses.info import Information
from zod.dataclasses.oxts import EgoMotion
from zod.dataclasses.sequence import ZodSequence
from zod.zod_sequences import ZodSequences


DATASET_ROOT = "/staging/dataset_donation/round_2/"
FILE_NAME = "ego_motion.json"
SEQUENCES = "sequences"
FRAMES = "single_frames"


def interpolate_and_write_oxts(info: Information, oxts: EgoMotion, save_dir: str):
    datetimes = [
        f.time
        for f in info.lidar_frames[Lidar.velodyne.value]
        + info.camera_frames["camera_front_original"]
    ]
    datetimes = sorted(list(set(datetimes)))

    # Use a set to remove duplicates
    timestamps = np.array({t.timestamp() for t in datetimes})

    ego_motion_interp = oxts.interpolate(timestamps)

    ego_motion_interp = ego_motion_interp.to_json()

    with open(osp.join(save_dir, info.id, "ego_motion.json"), "w") as f:
        json.dump(ego_motion_interp, f)


def process_frame(frame: ZodFrame, dataset_root: str):
    oxts = frame.get_oxts()
    interpolate_and_write_oxts(frame.info, oxts, save_dir=osp.join(dataset_root, FRAMES))


def process_sequence(sequence: ZodSequence, dataset_root: str):
    oxts = sequence.get_oxts()
    interpolate_and_write_oxts(sequence.info, oxts, save_dir=osp.join(dataset_root, SEQUENCES))


def main(
    dataset_root: str = typer.Option(DATASET_ROOT),
    sequences: bool = typer.Option(False),
    frames: bool = typer.Option(False),
):
    zod_sequences = ZodSequences(dataset_root=ROOT, version="full")
    # process_map(process_sequence, zod_sequences._sequences.values())
    for seq in zod_sequences:
        process_sequence(seq)


if __name__ == "__main__":
    typer.run(main)

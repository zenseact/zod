import os
import json
from tqdm.contrib.concurrent import process_map
import numpy as np
from zod.sequences.info import ZodSequenceInfo
from zod.sequences.zod_sequences import ZodSequences

ROOT = "/staging/dataset_donation/round_2"
SEQUENCES = "sequences"


def process_sequence(sequence: ZodSequenceInfo):
    ego_motion = sequence.get_ego_motion(from_json=True)

    datetimes = [
        f.time
        for f in sequence.lidar_frames["lidar_velodyne"]
        + sequence.camera_frames["camera_front_original"]
    ]
    datetimes = sorted(list(set(datetimes)))

    timestamps = np.array([t.timestamp() for t in datetimes])

    ego_motion_interp = ego_motion.interpolate(timestamps)

    ego_motion_interp = ego_motion_interp.to_json()

    # with open(os.path.join(ROOT, SEQUENCES, sequence.sequence_id, "ego_motion.json"), "w") as f:
    #    json.dump(ego_motion_interp, f)


def main():
    zod_sequences = ZodSequences(dataset_root=ROOT, version="full")
    # process_map(process_sequence, zod_sequences._sequences.values())
    for seq in zod_sequences._sequences.values():
        process_sequence(seq)


if __name__ == "__main__":
    main()

import json
import os
from datetime import datetime
from typing import Tuple

from dataclass_wizard import JSONWizard
from tqdm.contrib.concurrent import process_map

from zod.constants import DB_DATE_STRING_FORMAT_W_MICROSECONDS
from zod.dataclasses.zod_dataclasses import SensorFrame
from zod.sequences.info import ZodSequenceInfo


DATASET_ROOT = "/staging/dataset_donation/round_2/sequences"
FILE_NAME = "sequence_info.json"
SEQUENCES = "sequences"


def _split_filename(filename: str) -> Tuple[str, str, datetime]:
    seq_id, veh, timestamp = os.path.splitext(filename)[0].split("_")

    # parse timestamp 2022-02-14T13:23:33.029609Z
    timestamp = datetime.strptime(timestamp, DB_DATE_STRING_FORMAT_W_MICROSECONDS)

    return seq_id, veh, timestamp


class GlobalJSONMeta(JSONWizard.Meta):
    key_transform_with_dump = "SNAKE"


def create_sequence_info(sequence_folder: str) -> ZodSequenceInfo:
    sequence_path = os.path.join(DATASET_ROOT, sequence_folder)

    calibration_filename = os.listdir(os.path.join(sequence_path, "calibration"))[0]
    calibration_path = os.path.join(SEQUENCES, sequence_folder, "calibration", calibration_filename)

    oxts_filename = os.listdir(os.path.join(sequence_path, "oxts"))[0]
    oxts_path = os.path.join(SEQUENCES, sequence_folder, "oxts", oxts_filename)

    lidar_files = sorted(os.listdir(os.path.join(sequence_path, "lidar_velodyne")))
    lidar_frames = [
        SensorFrame(
            filepath=os.path.join(SEQUENCES, sequence_folder, "lidar_velodyne", filename),
            time=_split_filename(filename)[-1],
        )
        for filename in lidar_files
    ]

    camera_files = sorted(os.listdir(os.path.join(sequence_path, "camera_front_original")))
    camera_frames = [
        SensorFrame(
            filepath=os.path.join(SEQUENCES, sequence_folder, "camera_front_original", filename),
            time=_split_filename(filename)[-1],
        )
        for filename in camera_files
        if filename.endswith(".jpg")
    ]

    # TODO: fix this
    sequence_info = Information(
        sequence_id=sequence_folder,
        start_time=camera_frames[0].time,
        end_time=camera_frames[-1].time,
        camera_frames={"camera_front_original": camera_frames},
        lidar_frames={"lidar_velodyne": lidar_frames},
        oxts_path=oxts_path,
        calibration_path=calibration_path,
        metadata_path=os.path.join(SEQUENCES, sequence_folder, "metadata.json"),
        ego_motion_path=os.path.join(SEQUENCES, sequence_folder, "ego_motion.json"),
    )

    with open(os.path.join(sequence_path, FILE_NAME), "w") as f:
        json.dump(sequence_info.to_dict(), f)

    os.chmod(os.path.join(sequence_path, FILE_NAME), 0o664)

    return sequence_info


def main():
    folders = sorted(os.listdir(DATASET_ROOT))

    process_map(create_sequence_info, folders)


if __name__ == "__main__":
    main()
    main()

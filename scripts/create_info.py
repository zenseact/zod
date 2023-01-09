import glob
import json
import os
import os.path as osp
from datetime import datetime
from itertools import repeat
from typing import Dict, List, Tuple

import typer
from dataclass_wizard import JSONWizard
from tqdm.contrib.concurrent import process_map

from zod.constants import DB_DATE_STRING_FORMAT_W_MICROSECONDS
from zod.dataclasses.info import Information
from zod.dataclasses.zod_dataclasses import AnnotationFrame, SensorFrame


DATASET_ROOT = "/staging/dataset_donation/round_2/"
FILE_NAME = "info.json"
SEQUENCES = "sequences"
FRAMES = "single_frames"


def _split_filename(filename: str) -> Tuple[str, str, datetime]:
    seq_id, veh, timestamp = os.path.splitext(filename)[0].split("_")

    # parse timestamp 2022-02-14T13:23:33.029609Z
    timestamp = datetime.strptime(timestamp, DB_DATE_STRING_FORMAT_W_MICROSECONDS)

    return seq_id, veh, timestamp


class GlobalJSONMeta(JSONWizard.Meta):
    key_transform_with_dump = "SNAKE"


def create_sequence_info(id_: str, dataset_root: str, top_dir: str) -> Information:
    dir_path = os.path.join(dataset_root, top_dir, id_)

    calibration_filename = os.listdir(os.path.join(dir_path, "calibration"))[0]
    calibration_path = os.path.join(top_dir, id_, "calibration", calibration_filename)

    oxts_filename = os.listdir(os.path.join(dir_path, "oxts"))[0]
    oxts_path = os.path.join(top_dir, id_, "oxts", oxts_filename)

    lidar_frames = _get_lidar_frames(id_, top_dir, dir_path)

    all_camera_frames = _get_camera_frames(id_, top_dir, dir_path)

    all_annotation_frames = _get_annotation_frames(id_, top_dir, dir_path)

    # start time is the smallest timestamp of all camera frames or lidar frames
    # end time is the largest timestamp of all camera frames or lidar frames
    # keyframe time is the timestamp of the middle camera frame
    # assert equal number of frames for all cameras
    assert all(
        len(frames) == len(next(iter(all_camera_frames.values())))
        for frames in all_camera_frames.values()
    ), "Number of frames for all cameras should be equal"
    camera_times = [frame.time for frame in next(iter(all_camera_frames.values()))]
    lidar_times = [frame.time for frame in lidar_frames]

    sequence_info = Information(
        id=id_,
        start_time=min(camera_times + lidar_times),
        end_time=max(camera_times + lidar_times),
        keyframe_time=camera_times[len(camera_times) // 2],
        camera_frames=all_camera_frames,
        lidar_frames={"velodyne": lidar_frames},
        calibration_path=calibration_path,
        ego_motion_path=os.path.join(top_dir, id_, "ego_motion.json"),
        metadata_path=os.path.join(top_dir, id_, "metadata.json"),
        oxts_path=oxts_path,
        annotation_frames=all_annotation_frames,
    )

    with open(os.path.join(dir_path, FILE_NAME), "w") as f:
        json.dump(sequence_info.to_dict(), f)

    os.chmod(os.path.join(dir_path, FILE_NAME), 0o664)

    return sequence_info


def _get_lidar_frames(id_, top_dir, dir_path) -> List[SensorFrame]:
    lidar_files = sorted(os.listdir(os.path.join(dir_path, "lidar_velodyne")))
    lidar_frames = [
        SensorFrame(
            filepath=os.path.join(top_dir, id_, "lidar_velodyne", filename),
            time=_split_filename(filename)[-1],
        )
        for filename in lidar_files
    ]
    return lidar_frames


def _get_annotation_frames(id_, top_dir, dir_path) -> Dict[str, List[AnnotationFrame]]:
    if not os.path.exists(os.path.join(dir_path, "annotations")):
        return {}
    all_annotation_frames = {}
    for project in sorted(os.listdir(os.path.join(dir_path, "annotations"))):
        annotation_files = sorted(os.listdir(os.path.join(dir_path, "annotations", project)))
        annotation_frames = [
            AnnotationFrame(
                filepath=os.path.join(top_dir, id_, "annotations", project, filename),
                time=_split_filename(filename)[-1],
                project=project,
            )
            for filename in annotation_files
            if filename.endswith(".json")
        ]
        all_annotation_frames[project] = annotation_frames
    return all_annotation_frames


def _get_camera_frames(id_, top_dir, dir_path) -> Dict[str, List[SensorFrame]]:
    all_camera_frames = {}
    for camera in sorted(glob.glob(os.path.join(dir_path, "camera_*"))):
        camera_files = sorted(os.listdir(camera))

        camera_frames = [
            SensorFrame(
                filepath=os.path.join(top_dir, id_, osp.basename(camera), filename),
                time=_split_filename(filename)[-1],
            )
            for filename in camera_files
            if filename.endswith(".jpg")
        ]
        all_camera_frames[osp.basename(camera).lstrip("camera_")] = camera_frames
    return all_camera_frames


def main(
    dataset_root: str = typer.Option(DATASET_ROOT),
    sequences: bool = typer.Option(False),
    frames: bool = typer.Option(False),
):
    if not sequences and not frames:
        raise typer.BadParameter("Please specify either --sequences or --frames")
    if sequences:
        folders = sorted(os.listdir(osp.join(dataset_root, SEQUENCES)))
        process_map(
            create_sequence_info,
            folders,
            repeat(dataset_root),
            repeat(SEQUENCES),
        )
    if frames:
        folders = sorted(os.listdir(osp.join(dataset_root, FRAMES)))
        process_map(
            create_sequence_info,
            folders,
            repeat(dataset_root),
            repeat(FRAMES),
        )


if __name__ == "__main__":
    typer.run(main)

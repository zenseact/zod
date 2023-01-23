import glob
import json
import os
import os.path as osp
from datetime import datetime
from itertools import chain, repeat
from typing import Dict, List, Tuple

import typer
from dataclass_wizard import JSONWizard
from tqdm.contrib.concurrent import process_map

from zod.constants import (
    DB_DATE_STRING_FORMAT_W_MICROSECONDS,
    FRAMES,
    SEQUENCES,
    SPLIT_FILES,
    TRAIN,
    TRAINVAL_FILES,
    VAL,
    AnnotationProject,
    Anonymization,
    Lidar,
)
from zod.zod_dataclasses.info import Information
from zod.zod_dataclasses.sensor import AnnotationFrame, CameraFrame, LidarFrame

DATASET_ROOT = "/staging/dataset_donation/round_2/"
FILE_NAME = "info.json"


def _split_filename(filename: str) -> Tuple[str, str, datetime]:
    seq_id, veh, timestamp = osp.splitext(filename)[0].split("_")

    # parse timestamp 2022-02-14T13:23:33.029609Z
    timestamp = datetime.strptime(timestamp, DB_DATE_STRING_FORMAT_W_MICROSECONDS)

    return seq_id, veh, timestamp


class GlobalJSONMeta(JSONWizard.Meta):
    key_transform_with_dump = "SNAKE"


def create_sequence_info(id_: str, dataset_root: str, top_dir: str) -> Information:
    dir_path = osp.join(dataset_root, top_dir, id_)

    calibration_filename = os.listdir(osp.join(dir_path, "calibration"))[0]
    calibration_path = osp.join(top_dir, id_, "calibration", calibration_filename)

    oxts_filename = os.listdir(osp.join(dir_path, "oxts"))[0]
    oxts_path = osp.join(top_dir, id_, "oxts", oxts_filename)

    all_lidar_frames = _get_lidar_frames(id_, top_dir, dir_path)

    all_camera_frames = _get_camera_frames(id_, top_dir, dir_path)

    all_annotation_frames = _get_annotation_frames(id_, top_dir, dir_path)

    # start time is the smallest timestamp of all camera frames or lidar frames
    # end time is the largest timestamp of all camera frames or lidar frames
    # keyframe time is the timestamp of the middle camera frame
    # assert equal number of frames for all cameras
    assert all(
        len(frames) == len(next(iter(all_camera_frames.values())))
        for frames in all_camera_frames.values()
    ), f"Number of frames for all cameras should be equal, but are not for sequence {id_}"
    first_camera_times = [frame.time for frame in next(iter(all_camera_frames.values()))]
    all_times = {
        frame.time for frame in chain(*all_camera_frames.values(), *all_lidar_frames.values())
    }

    sequence_info = Information(
        id=id_,
        start_time=min(all_times),
        end_time=max(all_times),
        keyframe_time=first_camera_times[len(first_camera_times) // 2],
        camera_frames=all_camera_frames,
        lidar_frames=all_lidar_frames,
        calibration_path=calibration_path,
        ego_motion_path=osp.join(top_dir, id_, "ego_motion.json"),
        metadata_path=osp.join(top_dir, id_, "metadata.json"),
        oxts_path=oxts_path,
        annotation_frames=all_annotation_frames,
    )

    with open(osp.join(dir_path, FILE_NAME), "w") as f:
        json.dump(sequence_info.to_dict(), f)

    try:
        os.chmod(osp.join(dir_path, FILE_NAME), 0o664)
    except PermissionError:
        pass

    return sequence_info


def _get_lidar_frames(id_, top_dir, dir_path) -> Dict[Lidar, List[LidarFrame]]:
    all_lidar_frames = {}
    for lidar in Lidar:
        lidar_name = f"lidar_{lidar.value}"
        lidar_dir = osp.join(dir_path, lidar_name)
        if not osp.exists(lidar_dir):
            continue
        lidar_files = sorted(os.listdir(lidar_dir))
        lidar_frames = [
            LidarFrame(
                filepath=osp.join(top_dir, id_, lidar_name, filename),
                time=_split_filename(filename)[-1],
                is_compensated=top_dir == SEQUENCES,
            )
            for filename in lidar_files
        ]
        all_lidar_frames[lidar] = lidar_frames
    return all_lidar_frames


def _get_annotation_frames(
    id_, top_dir, dir_path
) -> Dict[AnnotationProject, List[AnnotationFrame]]:
    if not osp.exists(osp.join(dir_path, "annotations")):
        return {}
    all_annotation_frames = {}
    for project_name in sorted(os.listdir(osp.join(dir_path, "annotations"))):
        if project_name not in {p.value for p in AnnotationProject}:
            continue
        annotation_files = sorted(os.listdir(osp.join(dir_path, "annotations", project_name)))
        project = AnnotationProject(project_name)
        annotation_frames = [
            AnnotationFrame(
                filepath=osp.join(top_dir, id_, "annotations", project_name, filename),
                time=_split_filename(filename)[-1],
                project=project,
            )
            for filename in annotation_files
            if filename.endswith(".json")
        ]
        all_annotation_frames[project] = annotation_frames
    return all_annotation_frames


def _get_camera_frames(id_, top_dir, dir_path) -> Dict[str, List[CameraFrame]]:
    all_camera_frames = {}
    for camera in sorted(glob.glob(osp.join(dir_path, "camera_*"))):
        camera_files = sorted(os.listdir(camera))

        camera_frames = [
            CameraFrame(
                filepath=osp.join(top_dir, id_, osp.basename(camera), filename),
                time=_split_filename(filename)[-1],
            )
            for filename in camera_files
            if filename.endswith(".jpg")
        ]
        if camera_frames:
            all_camera_frames[osp.basename(camera).lstrip("camera_")] = camera_frames
    # TODO: this is temporary, remove
    blur = Anonymization.BLUR.value
    for camera, frames in list(all_camera_frames.items()):
        if blur in camera:
            continue
        blur_camera = "_".join(camera.split("_")[:-1] + [blur])
        if blur_camera in all_camera_frames:
            continue
        all_camera_frames[blur_camera] = [
            CameraFrame(
                filepath=frame.filepath.replace(camera, blur_camera),
                time=frame.time,
            )
            for frame in frames
        ]
    return all_camera_frames


def create_train_val_files(all_infos: List[Information], dataset_root: str, top_dir: str):
    infos = {info.id: info.to_dict() for info in all_infos}
    for version, train_val_file in TRAINVAL_FILES[top_dir].items():
        with open(osp.join(dataset_root, SPLIT_FILES[top_dir][version][TRAIN])) as f:
            train_ids = set(f.read().splitlines())
        with open(osp.join(dataset_root, SPLIT_FILES[top_dir][version][VAL])) as f:
            val_ids = set(f.read().splitlines())
        with open(osp.join(dataset_root, train_val_file), "w") as f:
            json.dump(
                {
                    TRAIN: [infos[id_] for id_ in train_ids],
                    VAL: [infos[id_] for id_ in val_ids],
                },
                f,
            )


def main(
    dataset_root: str = typer.Option(DATASET_ROOT),
    sequences: bool = typer.Option(False),
    frames: bool = typer.Option(False),
):
    if not sequences and not frames:
        raise typer.BadParameter("Please specify either --sequences or --frames")
    if sequences:
        folders = sorted(os.listdir(osp.join(dataset_root, SEQUENCES)))
        all_infos = process_map(
            create_sequence_info,
            folders,
            repeat(dataset_root),
            repeat(SEQUENCES),
            chunksize=10,
        )
        create_train_val_files(all_infos, dataset_root, SEQUENCES)
    if frames:
        folders = sorted(os.listdir(osp.join(dataset_root, FRAMES)))
        all_infos = process_map(
            create_sequence_info,
            folders,
            repeat(dataset_root),
            repeat(FRAMES),
            chunksize=100,
        )
        create_train_val_files(all_infos, dataset_root, FRAMES)


if __name__ == "__main__":
    typer.run(main)

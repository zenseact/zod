"""ZOD Frames module."""
import json
import os
from itertools import repeat
from pathlib import Path
from typing import Dict, List, Tuple, Union

import h5py
from tqdm.contrib.concurrent import process_map

from zod import constants
from zod.frames.annotation_parser import parse_ego_road_annotation, parse_lane_markings_annotation
from zod.frames.info import FrameInformation
from zod.utils.compensation import motion_compensate_pointwise, motion_compensate_scanwise
from zod.utils.objects import AnnotatedObject
from zod.utils.oxts import EgoMotion
from zod.utils.utils import zfill_id
from zod.utils.zod_dataclasses import Calibration, LidarData, MetaData


def _create_frame(frame: dict, dataset_root: str) -> FrameInformation:
    frame_info = FrameInformation.from_dict(frame)
    frame_info.convert_paths_to_absolute(dataset_root)
    return frame_info


def _get_lidar_frames_path(
    core_frame_path: str, n_sweeps_before: int, n_sweeps_after: int
) -> List[str]:
    if n_sweeps_before == 0 and n_sweeps_after == 0:
        return [core_frame_path]

    surrounding_frames = sorted(os.listdir(os.path.dirname(core_frame_path)))
    core_frame_idx = surrounding_frames.index(os.path.basename(core_frame_path))
    frame_paths = [
        os.path.join(os.path.dirname(core_frame_path), f)
        for f in surrounding_frames[
            max(0, core_frame_idx - n_sweeps_before) : core_frame_idx + n_sweeps_after + 1
        ]
    ]
    return frame_paths


class ZodFrames(object):
    """ZOD Frames."""

    def __init__(self, dataset_root: Union[Path, str], version: str):
        self._dataset_root = dataset_root
        self._version = version
        assert (
            version in constants.VERSIONS
        ), f"Unknown version: {version}, must be one of: {constants.VERSIONS}"
        self._train_frames, self._val_frames = self._load_frames()
        self._frames: Dict[str, FrameInformation] = {**self._train_frames, **self._val_frames}

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, frame_id: Union[int, str]) -> FrameInformation:
        """Get frame by id, which is zero-padded frame number."""
        frame_id = zfill_id(frame_id)
        return self._frames[frame_id]

    def _load_frames(self) -> Tuple[Dict[str, FrameInformation], Dict[str, FrameInformation]]:
        """Load frames for the given version."""
        filename = constants.SPLIT_TO_TRAIN_VAL_FILE_SINGLE_FRAMES[self._version]

        with open(os.path.join(self._dataset_root, filename), "r") as f:
            all_ids = json.load(f)

        train_frames = process_map(
            _create_frame,
            all_ids[constants.TRAIN],
            repeat(self._dataset_root),
            desc="Loading train frames",
            chunksize=50 if self._version == constants.FULL else 1,
        )
        val_frames = process_map(
            _create_frame,
            all_ids[constants.VAL],
            repeat(self._dataset_root),
            desc="Loading val frames",
            chunksize=50 if self._version == constants.FULL else 1,
        )

        train_frames = {frame.frame_id: frame for frame in train_frames}
        val_frames = {frame.frame_id: frame for frame in val_frames}

        return train_frames, val_frames

    def get_split(self, split: str) -> List[str]:
        """Get split by name (e.g. train / val)."""
        if split == constants.TRAIN:
            return list(self._train_frames.keys())
        elif split == constants.VAL:
            return list(self._val_frames.keys())
        else:
            raise ValueError(
                f"Unknown split: {split}, should be {constants.TRAIN} or {constants.VAL}"
            )

    def get_all_frames(self) -> Dict[str, FrameInformation]:
        """Get all frames."""
        return self._frames

    def get_camera_frame(
        self, frame_id: Union[int, str], anonymization_method: str = "blur", camera: str = "front"
    ) -> str:
        """Get camera frame with anonymization_method either "blur"
        or "dnat" from camera "front" or *not yet available*"""
        frame_id = zfill_id(frame_id)
        if anonymization_method not in ["blur", "dnat"]:
            raise ValueError("Not a valid anonymization method")
        return self._frames[frame_id].camera_frame["camera_" + camera + "_" + anonymization_method]

    def get_image_path(
        self, frame_id: Union[int, str], anonymization_method: str = "blur", camera: str = "front"
    ) -> str:
        """Get camera frame with anonymization_method either "blur"
        or "dnat" from camera "front" or *not yet available*"""
        if anonymization_method not in ["blur", "dnat"]:
            raise ValueError("Not a valid anonymization method")
        return (
            self._frames[zfill_id(frame_id)]
            .camera_frame["camera_" + camera + "_" + anonymization_method]
            .filepath
        )

    def get_lane_markings_annotation_path(self, frame_id: Union[int, str]) -> str:
        """Get the path to lane markings annotation"""
        return self._frames[zfill_id(frame_id)].lane_markings_annotation_path

    def get_ego_road_annotation_path(self, frame_id: Union[int, str]) -> str:
        """Get the path to ego road annotation"""
        return self._frames[zfill_id(frame_id)].ego_road_annotation_path

    def get_timestamp(self, frame_id: Union[int, str]):
        return self._frames[zfill_id(frame_id)].timestamp

    def read_lane_markings_annotation(self, frame_id: Union[int, str]):
        p = self.get_lane_markings_annotation_path(frame_id)
        return parse_lane_markings_annotation(p)

    def read_ego_road_annotation(self, frame_id: Union[int, str]):
        p = self.get_ego_road_annotation_path(frame_id)
        return parse_ego_road_annotation(p)

    def read_calibration(self, frame_id: str) -> Calibration:
        """Read calibration files from json format."""
        with open(self._frames[frame_id].calibration_path) as f:
            return Calibration.from_dict(json.load(f))

    def read_meta_data(self, frame_id: str) -> MetaData:
        """Read meta data files from json format."""
        with open(self._frames[frame_id].metadata_path) as f:
            meta_data = json.load(f)
        return MetaData.from_dict(meta_data)

    def read_ego_motion(self, frame_id: str) -> EgoMotion:
        """Read ego motion from npy format."""
        with h5py.File(self._frames[frame_id].oxts_path, "r") as f:
            return EgoMotion.from_frame_oxts(f)

    def read_object_detection_annotation(self, frame_id: str) -> List[AnnotatedObject]:
        """Read object detection annotation from json format."""
        with open(self._frames[frame_id].object_detection_annotation_path) as f:
            objs = json.load(f)
        return [AnnotatedObject.from_dict(anno) for anno in objs]

    def motion_compensate_pointcloud(
        self, lidar_data: LidarData, frame_id: str, lidar_name: str
    ) -> LidarData:
        return motion_compensate_pointwise(
            lidar_data,
            self.read_ego_motion(frame_id),
            self.read_calibration(frame_id).lidars[lidar_name],
        )

    def read_pointcloud(
        self,
        frame_id: str,
        lidar_name: str,
        n_sweeps_before: int = 0,
        n_sweeps_after: int = 0,
        motion_compensation: bool = False,
    ) -> LidarData:
        """Read pointcloud from npy format."""
        frame_paths = _get_lidar_frames_path(
            self._frames[frame_id].lidar_frame[lidar_name].filepath,
            n_sweeps_before,
            n_sweeps_after,
        )

        lidar_data = LidarData.empty()
        for i, frame_path in enumerate(frame_paths):
            lidar_data.append(LidarData.from_npy(frame_path))

        if motion_compensation:
            # TODO fix this
            lidar_data = motion_compensate_scanwise(lidar_data, frame_id, lidar_name)

        return lidar_data

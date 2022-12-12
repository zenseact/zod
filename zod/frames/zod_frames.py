"""ZOD Frames module."""
import json
import os
from itertools import repeat
from pathlib import Path
from typing import Dict, List, Tuple, Union

import h5py
from tqdm.contrib.concurrent import process_map

from zod.frames.annotation_parser import parse_ego_road_annotation, parse_lane_markings_annotation
from zod import constants
from zod.utils.objects import AnnotatedObject
from zod.utils.zod_dataclasses import Calibration, FrameInformation, LidarData, OXTSData


def _create_frame(frame: dict, dataset_root: str) -> FrameInformation:
    frame_info = FrameInformation.from_dict(frame)
    frame_info.convert_paths_to_absolute(dataset_root)
    return frame_info


def _get_frame_id(frame_id: Union[int, str]) -> str:
    if isinstance(frame_id, int):
        frame_id = str(frame_id).zfill(6)
    elif isinstance(frame_id, str):
        frame_id = frame_id.zfill(6)
    return frame_id


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
        frame_id = _get_frame_id(frame_id)
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
        frame_id = _get_frame_id(frame_id)
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
            self._frames[_get_frame_id(frame_id)]
            .camera_frame["camera_" + camera + "_" + anonymization_method]
            .filepath
        )

    def get_lane_markings_annotation_path(self, frame_id: Union[int, str]) -> str:
        """Get the path to lane markings annotation"""
        return self._frames[_get_frame_id(frame_id)].lane_markings_annotation_path

    def get_ego_road_annotation_path(self, frame_id: Union[int, str]) -> str:
        """Get the path to ego road annotation"""
        return self._frames[_get_frame_id(frame_id)].ego_road_annotation_path

    def get_timestamp(self, frame_id: Union[int, str]):
        return self._frames[_get_frame_id(frame_id)].timestamp

    def read_lane_markings_annotation(self, frame_id: Union[int, str]):
        p = self.get_lane_markings_annotation_path(frame_id)
        return parse_lane_markings_annotation(p)

    def read_ego_road_annotation(self, frame_id: Union[int, str]):
        p = self.get_ego_road_annotation_path(frame_id)
        return parse_ego_road_annotation(p)

    def read_calibration(self, frame_id: str) -> Calibration:
        """Read calibration files from json format."""
        with open(self._frames[frame_id].calibration_path) as f:
            calib = json.load(f)
        return Calibration.from_dict(calib)

    def read_oxts(self, frame_id: str) -> OXTSData:
        """Read OXTS files from hdf5 format."""
        with h5py.File(self._frames[frame_id].oxts_path, "r") as f:
            data = OXTSData.from_hdf5(f)
        return data

    def read_object_detection_annotation(self, frame_id: str) -> List[AnnotatedObject]:
        """Read object detection annotation from json format."""
        with open(self._frames[frame_id].object_detection_annotation_path) as f:
            objs = json.load(f)
        return [AnnotatedObject.from_dict(anno) for anno in objs]

    def read_pointcloud(
        self,
        frame_id: str,
        lidar_name: str,
        accumulate: bool = False,
        n_sweeps_before: int = 0,
        n_sweeps_after: int = 0,
    ) -> LidarData:
        """Read pointcloud from npy format."""
        if accumulate:
            raise NotImplementedError
        path = self._frames[frame_id].lidar_frame[lidar_name].filepath

        return LidarData.from_npy(path)

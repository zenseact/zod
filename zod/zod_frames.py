"""ZOD Frames module."""
import json
import os
from itertools import repeat
from pathlib import Path
from typing import Dict, List, Tuple, Union

from tqdm.contrib.concurrent import process_map

from zod import constants
from zod.frames.frame import ZodFrame
from zod.frames.info import FrameInfo
from zod.utils.compensation import motion_compensate_pointwise, motion_compensate_scanwise
from zod.utils.utils import zfill_id
from zod.utils.zod_dataclasses import LidarData


def _create_frame(frame: dict, dataset_root: str) -> FrameInfo:
    frame_info = FrameInfo.from_dict(frame)
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
        self._frames: Dict[str, FrameInfo] = {**self._train_frames, **self._val_frames}

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, frame_id: Union[int, str]) -> ZodFrame:
        """Get frame by id, which is zero-padded frame number."""
        frame_id = zfill_id(frame_id)
        return ZodFrame(self._frames[frame_id])

    def _load_frames(self) -> Tuple[Dict[str, FrameInfo], Dict[str, FrameInfo]]:
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

    def get_all_frame_infos(self) -> Dict[str, FrameInfo]:
        """Get all frames."""
        return self._frames

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

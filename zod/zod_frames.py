"""ZOD Frames module."""
import json
import os
import os.path as osp
from itertools import repeat
from pathlib import Path
from typing import Dict, List, Tuple, Union

from tqdm.contrib.concurrent import process_map

from zod import constants
from zod.utils.utils import zfill_id
from zod.zod_dataclasses.frame import ZodFrame
from zod.zod_dataclasses.info import Information


def _create_frame(frame: dict, dataset_root: str) -> Information:
    info = Information.from_dict(frame)
    info.convert_paths_to_absolute(dataset_root)
    return info


class ZodFrames(object):
    """ZOD Frames."""

    def __init__(self, dataset_root: Union[Path, str], version: str):
        self._dataset_root = dataset_root
        self._version = version
        assert (
            version in constants.VERSIONS
        ), f"Unknown version: {version}, must be one of: {constants.VERSIONS}"
        self._train_frames, self._val_frames = self._load_frames()
        self._frames: Dict[str, Information] = {**self._train_frames, **self._val_frames}

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, frame_id: Union[int, str]) -> ZodFrame:
        """Get frame by id, which is a 6-digit zero-padded number. Ex: '000001'."""
        frame_id = zfill_id(frame_id)
        return ZodFrame(self._frames[frame_id])

    def __iter__(self):
        for frame_id in self._frames:
            yield self.__getitem__(frame_id)

    def _load_frames(self) -> Tuple[Dict[str, Information], Dict[str, Information]]:
        """Load frames for the given version."""
        filename = constants.SPLIT_TO_TRAIN_VAL_FILE_SINGLE_FRAMES[self._version]

        with open(osp.join(self._dataset_root, filename), "r") as f:
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

        train_frames = {frame.id: frame for frame in train_frames}
        val_frames = {frame.id: frame for frame in val_frames}

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

    def get_all_frame_infos(self) -> Dict[str, Information]:
        """Get all frames."""
        return self._frames

    def get_all_ids(self) -> List[str]:
        """Get all frame ids."""
        return list(self._frames.keys())

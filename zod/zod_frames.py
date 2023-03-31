"""ZOD Frames module."""
import json
import os.path as osp
from functools import partial
from typing import Dict, List, Set, Tuple, Union

from tqdm.contrib.concurrent import process_map

from zod.constants import FRAMES, FULL, TRAIN, TRAINVAL_FILES, VAL, VERSIONS
from zod.data_classes.frame import ZodFrame
from zod.data_classes.info import Information
from zod.utils.utils import zfill_id


def _create_frame(frame: dict, dataset_root: str) -> Information:
    info = Information.from_dict(frame)
    info.convert_paths_to_absolute(dataset_root)
    return info


class ZodFrames:
    """ZOD Frames."""

    _TRAINVAL_FILES = TRAINVAL_FILES[FRAMES]

    def __init__(self, dataset_root: str, version: str, mp: bool = True):
        self._dataset_root = dataset_root
        self._version = version
        self._mp = mp
        assert version in VERSIONS, f"Unknown version: {version}, must be one of: {VERSIONS}"
        self._train_frames, self._val_frames = self._load_frames()
        self._frames: Dict[str, Information] = {**self._train_frames, **self._val_frames}

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, frame_id: Union[int, str, slice]) -> ZodFrame:
        """Get frame by id, which is a 6-digit zero-padded number. Ex: '000001'."""
        if isinstance(frame_id, slice):
            return [self.__getitem__(i) for i in range(*frame_id.indices(len(self)))]
        frame_id = zfill_id(frame_id)
        return ZodFrame(self._frames[frame_id])

    def __iter__(self):
        for frame_id in self._frames:
            yield self.__getitem__(frame_id)

    def _load_frames(self) -> Tuple[Dict[str, Information], Dict[str, Information]]:
        """Load frames for the given version."""
        filename = self._TRAINVAL_FILES[self._version]
        with open(osp.join(self._dataset_root, filename), "r") as f:
            all_ids = json.load(f)

        func = partial(_create_frame, dataset_root=self._dataset_root)
        if self._mp and self._version == FULL:
            train_frames = process_map(
                func,
                all_ids[TRAIN],
                desc="Loading train frames",
                chunksize=50 if self._version == FULL else 1,
            )
            val_frames = process_map(
                func,
                all_ids[VAL],
                desc="Loading val frames",
                chunksize=50 if self._version == FULL else 1,
            )
            train_frames = {frame.id: frame for frame in train_frames}
            val_frames = {frame.id: frame for frame in val_frames}
        else:
            train_frames = {frame.id: frame for frame in map(func, all_ids[TRAIN])}
            val_frames = {frame.id: frame for frame in map(func, all_ids[VAL])}

        return train_frames, val_frames

    def get_split(self, split: str) -> List[str]:
        """Get split by name (e.g. train / val)."""
        if split == TRAIN:
            return list(self._train_frames.keys())
        elif split == VAL:
            return list(self._val_frames.keys())
        else:
            raise ValueError(f"Unknown split: {split}, should be {TRAIN} or {VAL}")

    def get_all_infos(self) -> Dict[str, Information]:
        """Get all infos."""
        return self._frames

    def get_all_ids(self) -> Set[str]:
        """Get all frame ids."""
        return set(self._frames.keys())

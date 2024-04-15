"""ZOD Frames module."""

import json
import os.path as osp
from abc import ABC, abstractmethod
from functools import partial
from itertools import chain
from typing import Any, Dict, Optional, Set, Tuple, Union

from tqdm.contrib.concurrent import process_map

from zod.constants import FULL, TRAIN, VAL, VERSIONS, AnnotationProject, Split, Version
from zod.data_classes.info import Information
from zod.utils.utils import zfill_id


def _create_frame(frame: dict, dataset_root: str) -> Information:
    info = Information.from_dict(frame)
    info.convert_paths_to_absolute(dataset_root)
    return info


class ZodDataset(ABC):
    """Base class for all ZOD sub-datasets (frames, sequences, drives)."""

    def __init__(self, dataset_root: str, version: Version, mp: bool = True):
        self._dataset_root = dataset_root
        self._version = version
        self._mp = mp
        assert version in VERSIONS, f"Unknown version: {version}, must be one of: {VERSIONS}"
        self._infos, self._train_ids, self._val_ids = self._load_infos()

    def __len__(self) -> int:
        return len(self._infos)

    def __iter__(self):
        for frame_id in self._infos:
            if frame_id not in self._train_ids and frame_id not in self._val_ids:
                continue
            yield self.__getitem__(frame_id)

    @abstractmethod
    def __getitem__(self, frame_id: Union[int, str, slice]) -> Any:
        """Get item by id, which is a 6-digit zero-padded number. Ex: '000001'."""
        if isinstance(frame_id, slice):
            return [self.__getitem__(i) for i in range(*frame_id.indices(len(self)))]
        frame_id = zfill_id(frame_id)
        return self._infos[frame_id]

    @property
    @abstractmethod
    def trainval_files(self) -> Dict[str, str]:
        """Mapping of version to trainval file."""
        raise NotImplementedError

    def get_split(self, split: Split, project: Optional[AnnotationProject] = None) -> Set[str]:
        """Get split by name (e.g. train / val)."""
        if split == TRAIN:
            ids = self._train_ids
        elif split == VAL:
            ids = self._val_ids
        else:
            raise ValueError(f"Unknown split: {split}, should be {TRAIN} or {VAL}")

        if project is None:
            return ids
        else:
            return {id_ for id_ in ids if project in self._infos[id_].annotations}

    def get_all_infos(self) -> Dict[str, Information]:
        """Get all infos (including blacklisted ones)."""
        return self._infos

    def get_all_ids(self) -> Set[str]:
        """Get all frame ids (excluding blackisted ones)."""
        return self._train_ids.union(self._val_ids)

    def _load_infos(self) -> Tuple[Dict[str, Information], Set[str], Set[str]]:
        """Load frames for the given version."""
        trainval_path = osp.join(self._dataset_root, self.trainval_files[self._version])
        if not osp.exists(trainval_path):
            msg = f"Could not find trainval file: {trainval_path}.\n"
            if osp.exists(trainval_path.replace("-", "_")):
                msg += (
                    "However, found old, incompatible trainval files. Please either downgrade zod "
                    "to < 0.2 or download new files with `zod download --no-images --no-lidar`"
                )
            else:
                cls_name = f"{self.__class__.__name__}-{self._version}"
                msg += f"Make sure you have downloaded {cls_name} to {self._dataset_root}"
            raise FileNotFoundError(msg)
        with open(trainval_path, "r") as f:
            all_ids = json.load(f)

        func = partial(_create_frame, dataset_root=self._dataset_root)
        if self._mp and self._version == FULL:
            infos = process_map(
                func,
                chain.from_iterable(all_ids.values()),
                desc="Loading infos",
                chunksize=50 if self._version == FULL else 1,
            )
            infos = {frame.id: frame for frame in infos}
        else:
            infos = {frame.id: frame for frame in map(func, chain.from_iterable(all_ids.values()))}

        train_ids = {f["id"] for f in all_ids[TRAIN]}
        val_ids = {f["id"] for f in all_ids[VAL]}
        return infos, train_ids, val_ids

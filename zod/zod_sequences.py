import json
import os.path as osp
from functools import partial
from itertools import chain
from typing import Dict, List, Set, Tuple, Union

from tqdm.contrib.concurrent import process_map

from zod.constants import FULL, SEQUENCES, TRAIN, TRAINVAL_FILES, VAL, VERSIONS
from zod.data_classes.info import Information
from zod.data_classes.sequence import ZodSequence
from zod.utils.utils import zfill_id


def _create_sequence(sequence: dict, dataset_root: str) -> Information:
    info = Information.from_dict(sequence)
    info.convert_paths_to_absolute(dataset_root)
    return info


class ZodSequences:
    """ZOD Sequences."""

    _TRAINVAL_FILES = TRAINVAL_FILES[SEQUENCES]

    def __init__(self, dataset_root: str, version: str, mp: bool = True):
        self._dataset_root = dataset_root
        self._version = version
        self._mp = mp
        assert version in VERSIONS, f"Unknown version: {version}, must be one of: {VERSIONS}"
        self._sequences, self._train_ids, self._val_ids = self._load_sequences()

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, sequence_id: Union[int, str, slice]) -> ZodSequence:
        """Get sequence by id, which is a 6-digit zero-padded number. Ex: '000001'."""
        if isinstance(sequence_id, slice):
            return [self.__getitem__(i) for i in range(*sequence_id.indices(len(self)))]

        sequence_id = zfill_id(sequence_id)
        return ZodSequence(self._sequences[sequence_id])

    def __iter__(self):
        for frame_id in self._sequences:
            yield self.__getitem__(frame_id)

    def _load_sequences(self) -> Tuple[Dict[str, Information], List[str], List[str]]:
        """Load sequences for the given version."""
        filename = self._TRAINVAL_FILES[self._version]
        with open(osp.join(self._dataset_root, filename), "r") as f:
            all_ids = json.load(f)

        func = partial(_create_sequence, dataset_root=self._dataset_root)
        if self._mp and self._version == FULL:
            frames = process_map(
                func,
                chain.from_iterable(all_ids.values()),
                desc="Loading sequences",
                chunksize=50 if self._version == FULL else 1,
            )
            frames = {frame.id: frame for frame in frames}
        else:
            frames = {frame.id: frame for frame in map(func, chain.from_iterable(all_ids.values()))}

        train_ids = [f["id"] for f in all_ids[TRAIN]]
        val_ids = [f["id"] for f in all_ids[VAL]]
        return frames, train_ids, val_ids

    def get_split(self, split: str) -> List[str]:
        """Get split by name (e.g. train / val)."""
        if split == TRAIN:
            return self._train_ids
        elif split == VAL:
            return self._val_ids
        else:
            raise ValueError(f"Unknown split: {split}, should be {TRAIN} or {VAL}")

    def get_all_infos(self) -> Dict[str, Information]:
        """Get all infos."""
        return self._sequences

    def get_all_ids(self) -> Set[str]:
        """Get all sequence ids."""
        return set(self._sequences.keys())

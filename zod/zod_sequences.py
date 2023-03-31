import json
import os.path as osp
from functools import partial
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
        self._train_sequences, self._val_sequences = self._load_sequences()
        self._sequences: Dict[str, Information] = {
            **self._train_sequences,
            **self._val_sequences,
        }

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

    def _load_sequences(self) -> Tuple[Dict[str, Information], Dict[str, Information]]:
        """Load sequences for the given version."""
        filename = self._TRAINVAL_FILES[self._version]
        with open(osp.join(self._dataset_root, filename), "r") as f:
            all_ids = json.load(f)

        func = partial(_create_sequence, dataset_root=self._dataset_root)
        if self._mp and self._version == FULL:
            train_sequences = process_map(
                func,
                all_ids[TRAIN],
                desc="Loading train sequences",
                chunksize=10 if self._version == FULL else 1,
            )
            val_sequences = process_map(
                func,
                all_ids[VAL],
                desc="Loading val sequences",
                chunksize=10 if self._version == FULL else 1,
            )
            train_sequences = {s.id: s for s in train_sequences}
            val_sequences = {s.id: s for s in val_sequences}
        else:
            train_sequences = {s.id: s for s in map(func, all_ids[TRAIN])}
            val_sequences = {s.id: s for s in map(func, all_ids[VAL])}

        return train_sequences, val_sequences

    def get_split(self, split: str) -> List[str]:
        """Get split by name (e.g. train / val)."""
        if split == TRAIN:
            return list(self._train_sequences.keys())
        elif split == VAL:
            return list(self._val_sequences.keys())
        else:
            raise ValueError(f"Unknown split: {split}, should be {TRAIN} or {VAL}")

    def get_all_infos(self) -> Dict[str, Information]:
        """Get all infos."""
        return self._sequences

    def get_all_ids(self) -> Set[str]:
        """Get all sequence ids."""
        return set(self._sequences.keys())

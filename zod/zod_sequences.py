import json
import os.path as osp
from itertools import repeat
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union

from tqdm.contrib.concurrent import process_map

from zod.constants import SEQUENCES, TRAIN, TRAINVAL_FILES, VAL, VERSIONS
from zod.utils.utils import zfill_id
from zod.zod_dataclasses.info import Information
from zod.zod_dataclasses.sequence import ZodSequence


def _create_sequence(sequence: dict, dataset_root: str) -> Information:
    info = Information.from_dict(sequence)
    info.convert_paths_to_absolute(dataset_root)
    return info


class ZodSequences:
    """ZOD Sequences."""

    def __init__(self, dataset_root: Union[Path, str], version: str):
        self._dataset_root = dataset_root
        self._version = version
        assert version in VERSIONS, f"Unknown version: {version}, must be one of: {VERSIONS}"
        self._train_sequences, self._val_sequences = self._load_sequences()
        self._sequences: Dict[str, Information] = {
            **self._train_sequences,
            **self._val_sequences,
        }

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, sequence_id: Union[int, str]) -> ZodSequence:
        """Get sequence by id, which is a 6-digit zero-padded number. Ex: '000001'."""
        sequence_id = zfill_id(sequence_id)
        return ZodSequence(self._sequences[sequence_id])

    def __iter__(self):
        for frame_id in self._sequences:
            yield self.__getitem__(frame_id)

    def _load_sequences(self) -> Tuple[Dict[str, Information], Dict[str, Information]]:
        """Load sequences for the given version."""
        filename = TRAINVAL_FILES[SEQUENCES][self._version]

        with open(osp.join(self._dataset_root, filename), "r") as f:
            all_ids = json.load(f)

        train_sequences = process_map(
            _create_sequence,
            all_ids[TRAIN],
            repeat(self._dataset_root),
            chunksize=1,
            desc="Loading train sequences",
        )
        val_sequences = process_map(
            _create_sequence,
            all_ids[VAL],
            repeat(self._dataset_root),
            chunksize=1,
            desc="Loading val sequences",
        )

        train_sequences = {s.id: s for s in train_sequences}
        val_sequences = {s.id: s for s in val_sequences}

        return train_sequences, val_sequences

    def get_split(self, split: str) -> List[str]:
        """Get split by name (e.g. train / val)."""
        if split == TRAIN:
            return list(self._train_sequences.keys())
        elif split == VAL:
            return list(self._val_sequences.keys())
        else:
            raise ValueError(f"Unknown split: {split}, should be {TRAIN} or {VAL}")

    def get_all_ids(self) -> Set[str]:
        """Get all sequence ids."""
        return set(self._sequences.keys())

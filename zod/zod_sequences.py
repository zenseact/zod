import json
import os.path as osp
from itertools import repeat
from pathlib import Path
from typing import Dict, List, Union

from tqdm.contrib.concurrent import process_map

from zod import constants
from zod.dataclasses.info import Information
from zod.dataclasses.sequence import ZodSequence
from zod.utils.utils import zfill_id


def _create_sequence(sequence: dict, dataset_root: str) -> Information:
    info = Information.from_dict(sequence)
    info.convert_paths_to_absolute(dataset_root)
    return info


class ZodSequences:
    def __init__(self, dataset_root: Union[Path, str], version: str):
        self._dataset_root = dataset_root
        self._version = version
        assert (
            version in constants.VERSIONS
        ), f"Unknown version: {version}, must be one of: {constants.VERSIONS}"
        self._train_sequences, self._val_sequences = self._load_sequences()
        self._sequences: Dict[str, Information] = {
            **self._train_sequences,
            **self._val_sequences,
        }

    def __getitem__(self, sequence_id: Union[int, str]) -> ZodSequence:
        """Get sequence by id, which is zero-padded number."""
        sequence_id = zfill_id(sequence_id)
        return ZodSequence(self._sequences[sequence_id])

    def __len__(self) -> int:
        return len(self._sequences)

    def __iter__(self):
        for frame_id in self._sequences:
            yield self.__getitem__(frame_id)

    def _load_sequences(self):
        filename = constants.SPLIT_TO_TRAIN_VAL_FILE_SEQUENCES[self._version]

        with open(osp.join(self._dataset_root, filename), "r") as f:
            all_ids = json.load(f)

        train_sequences = process_map(
            _create_sequence,
            all_ids[constants.TRAIN],
            repeat(self._dataset_root),
            chunksize=1,
            desc="Loading train sequences",
        )
        val_sequences = process_map(
            _create_sequence,
            all_ids[constants.VAL],
            repeat(self._dataset_root),
            chunksize=1,
            desc="Loading val sequences",
        )

        train_sequences = {s.id: s for s in train_sequences}
        val_sequences = {s.id: s for s in val_sequences}

        return train_sequences, val_sequences

    def get_split(self, split: str) -> List[str]:
        """Get split by name (e.g. train / val)."""
        if split == constants.TRAIN:
            return list(self._train_sequences.keys())
        elif split == constants.VAL:
            return list(self._train_sequences.keys())
        else:
            raise ValueError(
                f"Unknown split: {split}, should be {constants.TRAIN} or {constants.VAL}"
            )

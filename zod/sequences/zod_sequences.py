import json
import os
import os.path as osp
from itertools import repeat
from pathlib import Path
from typing import Dict, Union

from tqdm.contrib.concurrent import process_map

from zod import constants
from zod.sequences.info import SequenceInformation
from zod.sequences.zod_sequence import ZodSequence
from zod.utils.utils import zfill_id


def _create_sequence(sequence_folder: str, dataset_root: str) -> SequenceInformation:
    with open(osp.join(sequence_folder, "sequence_info.json"), "r") as f:
        sequence_info = json.load(f)

    sequence_info = SequenceInformation.from_dict(sequence_info)
    sequence_info.convert_paths_to_absolute(dataset_root)
    return sequence_info


class ZodSequences:
    def __init__(self, dataset_root: Union[Path, str], version: str):
        self._dataset_root = dataset_root
        self._version = version
        assert (
            version in constants.VERSIONS
        ), f"Unknown version: {version}, must be one of: {constants.VERSIONS}"
        self._train_sequences, self._val_sequences = self._load_sequences()
        self._sequences: Dict[str, SequenceInformation] = {
            **self._train_sequences,
            **self._val_sequences,
        }

    def _load_sequences(self):
        sequence_folder = osp.join(self._dataset_root, "sequences")
        folders = [osp.join(sequence_folder, f) for f in os.listdir(sequence_folder)]

        sequences = process_map(
            _create_sequence,
            folders,
            repeat(self._dataset_root),
            chunksize=1,
            desc="Loading sequences",
        )

        # TODO: fix this
        return {s.sequence_id: s for s in sequences[:900]}, {
            s.sequence_id: s for s in sequences[900:]
        }

    def __getitem__(self, sequence_id: Union[int, str]) -> ZodSequence:
        """Get sequence by id, which is zero-padded number."""
        sequence_id = zfill_id(sequence_id)
        return ZodSequence(self._sequences[sequence_id])

    def __len__(self) -> int:
        return len(self._sequences)

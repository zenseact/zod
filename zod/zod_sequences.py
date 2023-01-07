import os
import os.path as osp
from itertools import repeat
from pathlib import Path
from typing import Dict, Union

from tqdm.contrib.concurrent import process_map

from zod import constants
from zod.dataclasses.info import Information
from zod.dataclasses.sequence import ZodSequence
from zod.utils.utils import zfill_id


def _create_sequence(sequence_folder: str, dataset_root: str) -> Information:
    sequence_info = Information.from_json_path(osp.join(sequence_folder, "info.json"))
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

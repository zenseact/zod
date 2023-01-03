from pathlib import Path
from typing import Dict, Union

from zod import constants
from zod.sequences.info import SequenceInformation
from zod.utils.utils import zfill_id


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

    def _load_sequences():
        raise NotImplementedError

    def __getitem__(self, sequence_id: Union[int, str]) -> SequenceInformation:
        """Get sequence by id, which is zero-padded number."""
        sequence_id = zfill_id(sequence_id)
        return self._sequences[sequence_id]

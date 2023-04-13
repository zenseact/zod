from typing import Dict, Union

from zod._zod_dataset import ZodDataset
from zod.constants import SEQUENCES, TRAINVAL_FILES
from zod.data_classes.sequence import ZodSequence


class ZodSequences(ZodDataset):
    """ZOD Sequences."""

    def __getitem__(self, frame_id: Union[int, str, slice]) -> ZodSequence:
        """Get frame by id, which is a 6-digit zero-padded number. Ex: '000001'."""
        info = super().__getitem__(frame_id)
        return ZodSequence(info)

    @property
    def trainval_files(self) -> Dict[str, str]:
        return TRAINVAL_FILES[SEQUENCES]

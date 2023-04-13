"""ZOD Frames module."""
from typing import Dict, Union

from zod._zod_dataset import ZodDataset
from zod.constants import FRAMES, TRAINVAL_FILES
from zod.data_classes.frame import ZodFrame


class ZodFrames(ZodDataset):
    """ZOD Frames."""

    def __getitem__(self, frame_id: Union[int, str, slice]) -> ZodFrame:
        """Get frame by id, which is a 6-digit zero-padded number. Ex: '000001'."""
        info = super().__getitem__(frame_id)
        return ZodFrame(info)

    @property
    def trainval_files(self) -> Dict[str, str]:
        return TRAINVAL_FILES[FRAMES]

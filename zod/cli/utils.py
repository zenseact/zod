from enum import Enum
from typing import Type

from zod import ZodDataset, ZodDrives, ZodFrames, ZodSequences


class SubDataset(Enum):
    FRAMES = "frames"
    SEQUENCES = "sequences"
    DRIVES = "drives"

    @property
    def folder(self) -> str:
        if self == SubDataset.FRAMES:
            return "single_frames"
        else:
            return self.value

    @property
    def dataset_cls(self) -> Type[ZodDataset]:
        return _cls_map[self]


_cls_map = {
    SubDataset.FRAMES: ZodFrames,
    SubDataset.SEQUENCES: ZodSequences,
    SubDataset.DRIVES: ZodDrives,
}


class Version(Enum):
    FULL = "full"
    MINI = "mini"

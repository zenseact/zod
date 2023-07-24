import enum

from zod import ZodDrives, ZodFrames, ZodSequences


class SubDataset(enum.Enum):
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
    def dataset_cls(self):
        return _cls_map[self]


_cls_map = {
    SubDataset.FRAMES: ZodFrames,
    SubDataset.SEQUENCES: ZodSequences,
    SubDataset.DRIVES: ZodDrives,
}

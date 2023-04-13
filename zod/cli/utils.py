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
        if self == SubDataset.FRAMES:
            return ZodFrames
        elif self == SubDataset.SEQUENCES:
            return ZodSequences
        elif self == SubDataset.DRIVES:
            return ZodDrives
        else:
            raise ValueError(f"Unknown subdataset: {self}")

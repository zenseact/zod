"""ZOD Frames module."""

from collections import defaultdict
from typing import Dict, Union

from tqdm import tqdm

from zod._zod_dataset import ZodDataset
from zod.constants import FRAMES, TRAINVAL_FILES, AnnotationProject
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

    def get_subclass_counts(self, require_3d=False) -> Dict[str, int]:
        """Will scan the dataset and return all subclasses and their counts."""
        subclasses = defaultdict(int)
        for frame in tqdm(self, desc="Processing annotations..."):
            for obj in frame.get_annotation(AnnotationProject.OBJECT_DETECTION):
                if require_3d and obj.box3d is None:
                    continue
                subclasses[obj.subclass] += 1
        return subclasses

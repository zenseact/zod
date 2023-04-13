"""Top-level package for Zenseact Open Dataset (ZOD)."""

from ._zod_dataset import ZodDataset as ZodDataset  # For type hinting
from .constants import AnnotationProject as AnnotationProject
from .constants import Anonymization as Anonymization
from .constants import Camera as Camera
from .constants import Lidar as Lidar
from .data_classes.frame import ZodFrame as ZodFrame
from .data_classes.sequence import ZodSequence as ZodSequence
from .zod_drives import ZodDrives as ZodDrives
from .zod_frames import ZodFrames as ZodFrames
from .zod_sequences import ZodSequences as ZodSequences

try:
    # importlib.metadata is present in Python 3.8 and later
    import importlib.metadata as importlib_metadata
except ImportError:
    # use the shim package importlib-metadata pre-3.8
    import importlib_metadata as importlib_metadata

try:
    __version__ = importlib_metadata.version(__package__ or __name__)
except importlib_metadata.PackageNotFoundError:
    __version__ = "0.0.0"

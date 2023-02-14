"""Top-level package for Zenseact Open Dataset (ZOD)."""

from .constants import AnnotationProject as AnnotationProject
from .constants import Anonymization as Anonymization
from .constants import Camera as Camera
from .constants import Lidar as Lidar
from .zod_frames import ZodFrames as ZodFrames
from .zod_sequences import ZodSequences as ZodSequences

try:
    # importlib.metadata is present in Python 3.8 and later
    import importlib.metadata as importlib_metadata
except ImportError:
    # use the shim package importlib-metadata pre-3.8
    import importlib_metadata as importlib_metadata

__version__ = importlib_metadata.version(__package__ or __name__)

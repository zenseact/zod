"""Top-level package for Zenseact Open Dataset (ZOD)."""

import importlib.metadata as importlib_metadata

from .constants import AnnotationProject as AnnotationProject
from .constants import Anonymization as Anonymization
from .constants import Camera as Camera
from .constants import Lidar as Lidar
from .zod_frames import ZodFrames as ZodFrames
from .zod_sequences import ZodSequences as ZodSequences

__version__ = importlib_metadata.version(__package__ or __name__)

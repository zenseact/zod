"""Top-level package for Zenseact Open Dataset (ZOD)."""

from .frames.zod_frames import ZodFrames as ZodFrames


try:
    # importlib.metadata is present in Python 3.8 and later
    import importlib.metadata as importlib_metadata
except ImportError:
    # use the shim package importlib-metadata pre-3.8
    import importlib_metadata as importlib_metadata

try:
    # __package__ allows for the case where __name__ is "__main__"
    __version__ = importlib_metadata.version(__package__ or __name__)
except importlib_metadata.PackageNotFoundError:
    __version__ = "0.0.0"

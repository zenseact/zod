"""Utility functions for Zod."""

import os
from datetime import datetime
from typing import Union

from zod.constants import DB_DATE_STRING_FORMAT_W_MICROSECONDS


def datetime_from_str(timestamp: str) -> datetime:
    """Convert a timestamp string to a datetime object."""
    return datetime.strptime(timestamp, DB_DATE_STRING_FORMAT_W_MICROSECONDS)


def str_from_datetime(timestamp: datetime) -> str:
    """Convert a datetime object to a timestamp string."""
    return timestamp.strftime(DB_DATE_STRING_FORMAT_W_MICROSECONDS)


def parse_datetime_from_filename(filename: str) -> datetime:
    """Parse a timestamp from a filename."""
    timestamp_str = os.path.splitext(os.path.basename(filename))[0].split("_")[-1]
    return datetime_from_str(timestamp_str)


def zfill_id(frame_id: Union[int, str]) -> str:
    if isinstance(frame_id, int):
        frame_id = str(frame_id)
    return frame_id.zfill(6)

"""Utility functions for Zod."""

import os
from datetime import datetime, timedelta

from zod.constants import DB_DATE_STRING_FORMAT_W_MICROSECONDS, MICROSEC_PER_SEC

def datetime_from_str(timestamp: str) -> datetime:
    """Convert a timestamp string to a datetime object."""
    return datetime.strptime(timestamp, DB_DATE_STRING_FORMAT_W_MICROSECONDS)


def parse_timestamp_from_filename(filename: str) -> datetime:
    """Parse a timestamp from a filename."""
    timestamp_str = os.path.splitext(os.path.basename(filename))[0].split("_")[-1]
    return datetime_from_str(timestamp_str)

def gps_time_to_datetime(gps_time: float, leap_seconds: int = -18) -> datetime:
    """Convert GPS time to datetime."""

    return datetime(1980, 1, 6) + timedelta(microseconds=gps_time + leap_seconds * MICROSEC_PER_SEC)
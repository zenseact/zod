"""Utility functions for Zod."""

import os
import datetime


def datetime_from_str(timestamp: str) -> datetime.datetime:
    """Convert a timestamp string to a datetime object."""
    return datetime.datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")


def parse_timestamp_from_filename(filename: str) -> datetime.datetime:
    """Parse a timestamp from a filename."""
    timestamp_str = os.path.splitext(os.path.basename(filename))[0].split("_")[-1]
    return datetime_from_str(timestamp_str)

"""Backwards compatibility for oxts.py."""

import warnings

from .ego_motion import *

warnings.warn(
    "The 'zod.data_classes.oxts' module has been deprecated and will be removed in a future "
    "version. Please use the 'zod.data_classes.ego_motion' module instead.",
    DeprecationWarning,
    stacklevel=2,
)

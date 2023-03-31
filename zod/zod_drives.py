from zod.constants import DRIVES, TRAINVAL_FILES
from zod.zod_sequences import ZodSequences


class ZodDrives(ZodSequences):
    """ZOD Drives.

    Drives are fundamentally the same as sequences, just longer.
    Therefore

    """

    _TRAINVAL_FILES = TRAINVAL_FILES[DRIVES]

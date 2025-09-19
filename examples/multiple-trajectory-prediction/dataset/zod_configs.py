"""Global static parameters dataclass."""

from dataclasses import dataclass


@dataclass
class ZodConfigs:
    """ZOD configs class."""

    NUM_OUTPUT: int = 51
    IMG_SIZE: int = 256
    TARGET_DISTANCES: tuple = (
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        40,
        50,
        60,
        70,
        80,
        95,
        110,
        125,
        145,
        165,
    )

    BATCH_SIZE: int = 32
    TEST_SIZE: int = 0.1  # fraction of test data to use
    VAL_SIZE: int = 0.1  # fraction of train data to use for validation

    # File paths
    STORED_GROUND_TRUTH_PATH: str = "/mnt/ZOD/ground_truth.json"
    DATASET_ROOT: str = "/mnt/ZOD"

    USE_PRE_RESIZED_IMGS: bool = True

    NORMALIZE_MEAN: tuple = (0.485, 0.456, 0.406)
    NORMALIZE_STD: tuple = (0.229, 0.224, 0.225)

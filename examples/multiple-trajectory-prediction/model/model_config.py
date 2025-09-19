"""Model specific static parameters dataclass."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Global configs class."""

    LEARNING_RATE: float = 0.001
    TARGET_DISTANCES: list = (
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
    USE_GPU: bool = True
    NR_OF_MODES: int = 2
    EPOCHS = 10

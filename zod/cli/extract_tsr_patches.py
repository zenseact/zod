"""Script that prepares the dataset for the traffic sign classification task.

Will create a dataset from the original zod frames by cropping out the relevant traffic signs
in each image and storing them for later use.
"""

import json
import os
from dataclasses import dataclass
from itertools import repeat
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import typer
from PIL import Image
from tqdm.contrib.concurrent import process_map

import zod.anno.parser as parser
import zod.constants as constants
from zod import ZodFrames
from zod.cli.utils import Version
from zod.data_classes.frame import ZodFrame

SUMMARY = """
Finished creating dataset.
.....................................
Created {} traffic sign samples.
Saved dataset to {}, with a dataset.json file that contains the metadata.

Each image is padded with a factor of {}. Note that the padding might be limited
    by the original image size. The padding is stored in the dataset.json file.

Images are saved in the following folder structure:
    <output_folder>/<traffic_sign_class>/<frame_id>_<annotation_uuid>.png
"""


@dataclass
class Settings:
    output_dir: Path
    padding_factor: Optional[float]
    padding_px: Optional[Tuple[int, int]]
    overwrite: bool
    exclude_unclear: bool


def extract_tsr_patches(
    dataset_root: Path = typer.Option(
        ...,
        exists=True,
        dir_okay=True,
        writable=False,
        readable=True,
        resolve_path=True,
        help="Path to the root of the ZOD dataset.",
    ),
    output_dir: Path = typer.Option(
        ...,
        exists=False,
        file_okay=False,
        dir_okay=True,
        writable=True,
        readable=True,
        resolve_path=True,
        help="Path to the output directory.",
    ),
    version: str = typer.Option("full", help="Version of the dataset to use. One of: full, small."),
    padding_factor: Optional[float] = typer.Option(None, help="Factor to multiply the padding with."),
    padding_px_y: Optional[int] = typer.Option(None, help="Padding in y direction."),
    padding_px_x: Optional[int] = typer.Option(None, help="Padding in x direction."),
    num_workers: Optional[int] = typer.Option(None, help="Number of workers to use."),
    overwrite: bool = typer.Option(False, help="Whether to overwrite existing files."),
    exclude_unclear: bool = typer.Option(False, help="Whether to exclude unclear traffic signs."),
):
    """Create a dataset for traffic sign classification."""
    print("Will create dataset from", dataset_root)
    print("Will save dataset to", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    assert not (
        padding_factor is not None and (padding_px_x is not None or padding_px_y is not None)
    ), "Cannot specify both padding and padding_factor"

    padding = (padding_px_x, padding_px_y) if (padding_px_x is not None and padding_px_y is not None) else None

    settings = Settings(
        output_dir=output_dir,
        padding_factor=padding_factor,
        padding_px=padding,
        overwrite=overwrite,
        exclude_unclear=exclude_unclear,
    )

    zod_frames = ZodFrames(str(dataset_root), version=version.value)
    print(f"Will process {len(zod_frames)} full frames.")

    results = process_map(
        _process_frame,
        zod_frames,
        repeat(settings),
        repeat(zod_frames.get_split(constants.TRAIN)),
        desc="Processing frames in ZOD",
        max_workers=num_workers,
        chunksize=1 if num_workers == 1 else 100,
    )
    # flatten the returned list
    traffic_sign_frames: List[Dict[str, Any]] = [frame for frames in results for frame in frames]

    train_frame_ids = set(zod_frames.get_split(constants.TRAIN))
    val_frame_ids = set(zod_frames.get_split(constants.VAL))

    train_frames = [f for f in traffic_sign_frames if f["frame_id"].split("_")[0] in train_frame_ids]
    val_frames = [f for f in traffic_sign_frames if f["frame_id"].split("_")[0] in val_frame_ids]

    # write it to a json file
    with open(os.path.join(output_dir, "dataset.json"), "w") as f:
        json.dump({constants.TRAIN: train_frames, constants.VAL: val_frames}, f)

    print(
        SUMMARY.format(
            len(traffic_sign_frames),
            output_dir,
            padding_factor if padding_factor is not None else padding,
        )
    )


def _process_frame(frame: ZodFrame, settings: Settings, train_ids: Set[str]) -> List[Dict[str, Any]]:
    """Process a single frame."""

    # not all frames have traffic signs
    if constants.AnnotationProject.TRAFFIC_SIGNS not in frame.info.annotations:
        return []

    traffic_signs: List[parser.TrafficSignAnnotation] = frame.get_annotation(constants.AnnotationProject.TRAFFIC_SIGNS)

    if len(traffic_signs) == 0:
        return []

    new_cropped_frames = []
    # load the image
    image_path = frame.info.get_key_camera_frame(constants.Anonymization.BLUR).filepath
    image = np.array(Image.open(image_path))
    for traffic_sign in traffic_signs:
        if settings.exclude_unclear and traffic_sign.unclear:
            continue
        cls_name = "unclear" if traffic_sign.unclear else traffic_sign.traffic_sign_class
        train_or_val = constants.TRAIN if frame.info.id in train_ids else constants.VAL
        cls_folder = settings.output_dir / train_or_val / cls_name
        cls_folder.mkdir(parents=True, exist_ok=True)

        new_frame_id = f"{frame.info.id}_{traffic_sign.uuid}"
        output_file = os.path.join(
            cls_folder,
            f"{new_frame_id}.png",
        )

        # crop out the correct patch
        cropped_image, padding = traffic_sign.bounding_box.crop_from_image(
            image,
            padding_factor=settings.padding_factor,
            padding=settings.padding_px,
        )

        if settings.overwrite or not os.path.exists(output_file):
            pil_image = Image.fromarray(cropped_image)
            pil_image.save(output_file)
        original_widht = traffic_sign.bounding_box.dimension[0]
        original_height = traffic_sign.bounding_box.dimension[1]
        center_x = padding[0] + original_widht // 2
        center_y = padding[1] + original_height // 2

        # create a frame info file
        new_cropped_frames.append(
            {
                "frame_id": new_frame_id,
                "image_path": os.path.join(cls_folder, f"{new_frame_id}.png"),
                "padding_left": float(padding[0]),
                "padding_top": float(padding[1]),
                "padding_right": float(padding[2]),
                "padding_bottom": float(padding[3]),
                "padded_width": cropped_image.shape[1],
                "padded_height": cropped_image.shape[0],
                "center_x": float(center_x),
                "center_y": float(center_y),
                "original_width": float(traffic_sign.bounding_box.dimension[0]),
                "original_height": float(traffic_sign.bounding_box.dimension[1]),
                "annotation": {key: val for key, val in traffic_sign.__dict__.items() if key != "bounding_box"},
            }
        )

    return new_cropped_frames


if __name__ == "__main__":
    typer.run(extract_tsr_patches)

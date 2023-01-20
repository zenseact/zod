"""This module will generate a COCO JSON file from the ZOD dataset."""

import json
import os
from functools import partial
from pathlib import Path
from typing import List, Tuple

import typer
from tqdm.contrib.concurrent import process_map

import zod.constants as constants
from zod import ZodFrames
from zod.constants import ALL_CLASSES, Anonymization
from zod.utils.utils import str_from_datetime
from zod.zod_dataclasses.frame import ZodFrame
from zod.zod_dataclasses.objects import AnnotatedObject

# Map classes to categories, starting from 1
CATEGORY_NAME_TO_ID = {cls: i + 1 for i, cls in enumerate(ALL_CLASSES)}
OPEN_DATASET_URL = (
    "https://www.ai.se/en/data-factory/datasets/data-factory-datasets/zenseact-open-dataset"
)


def _convert_frame(
    frame: ZodFrame, classes: List[str], anonymization: constants.Anonymization, use_png: bool
) -> Tuple[dict, List[dict]]:

    objs: List[AnnotatedObject] = frame.get_annotation(constants.AnnotationProject.OBJECT_DETECTION)
    camera_frame = frame.info.get_key_camera_frame(anonymization=anonymization)
    file_name = camera_frame.filepath

    if anonymization == constants.Anonymization.ORIGINAL:
        file_name = file_name.replace(
            Anonymization.BLUR.value, constants.Anonymization.ORIGINAL.value
        )
    if use_png:
        file_name = file_name.replace(".jpg", ".png")

    image_dict = {
        "id": int(frame.info.id),
        "license": 1,
        "file_name": file_name,
        "height": camera_frame.height,
        "width": camera_frame.width,
        "date_captured": str_from_datetime(frame.info.keyframe_time),
    }
    anno_dicts = [
        {
            "id": int(frame.info.id) * 1000
            + obj_idx,  # avoid collisions by assuming max 1k objects per frame
            "image_id": int(frame.info.id),
            "category_id": CATEGORY_NAME_TO_ID[obj.name],
            "bbox": [round(val, 2) for val in obj.box2d.xywh.tolist()],
            "area": round(obj.box2d.area, 2),
            "iscrowd": obj.should_ignore_object(require_3d=False, require_eval=False),
        }
        for obj_idx, obj in enumerate(objs)
        if obj.name in classes
    ]
    return image_dict, anno_dicts


def generate_coco_json(
    dataset: ZodFrames,
    split: str,
    classes: List[str],
    anonymization: constants.Anonymization,
    use_png: bool,
) -> dict:
    """Generate COCO JSON file from the ZOD dataset."""
    assert split in ["train", "val"], f"Unknown split: {split}"
    frame_infos = [dataset[frame_id] for frame_id in dataset.get_split(split)]
    _convert_frame_w_classes = partial(
        _convert_frame, classes=classes, anonymization=anonymization, use_png=use_png
    )
    results = process_map(
        _convert_frame_w_classes,
        frame_infos,
        desc=f"Converting {split} frames",
        chunksize=50 if dataset._version == "full" else 1,
    )
    image_dicts, all_annos = zip(*results)
    anno_dicts = [anno for annos in all_annos for anno in annos]  # flatten
    coco_json = {
        "images": image_dicts,
        "annotations": anno_dicts,
        "info": {
            "description": "Zenseact Open Dataset",
            "url": OPEN_DATASET_URL,
            "version": dataset._version,  # TODO: add dataset versioning
            "year": 2022,
            "contributor": "ZOD team",
            "date_created": "2022/12/15",
        },
        "licenses": [
            {
                "url": "https://creativecommons.org/licenses/by-sa/4.0/",
                "name": "Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)",
                "id": 1,
            },
        ],
        "categories": [
            {"supercategory": "object", "id": category_id, "name": category_name}
            for category_name, category_id in CATEGORY_NAME_TO_ID.items()
            if category_name in classes
        ],
    }
    return coco_json


# Use typer instead of argparse
def convert_to_coco(
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
        exists=True,
        dir_okay=True,
        writable=True,
        readable=True,
        resolve_path=True,
        help="Path to the output directory.",
    ),
    version: str = typer.Option("full", help="Version of the dataset to use. One of: full, mini."),
    anonymization: constants.Anonymization = typer.Option(
        constants.Anonymization.BLUR, help="Anonymization mode to use."
    ),
    use_png: bool = typer.Option(False, help="Whether to use PNG images instead of JPG."),
    classes: List[str] = typer.Option(
        ["Vehicle", "Pedestrian", "VulnerableVehicle"], help="Classes to include in the dataset."
    ),
):
    # check that classes are valid
    for cls in classes:
        if cls not in ALL_CLASSES:
            typer.echo(f"ERROR: Invalid class: {cls}.")
            raise typer.Exit(1)
    typer.echo(
        "Converting ZOD to COCO format. "
        f"Version: {version}, anonymization: {anonymization}, classes: {classes}"
    )

    zod_frames = ZodFrames(dataset_root, version)

    base_name = f"zod_{version}_{anonymization}"
    if use_png:
        base_name += "_png"

    os.makedirs(output_dir, exist_ok=True)

    coco_json_train = generate_coco_json(
        zod_frames, split="train", classes=classes, anonymization=anonymization, use_png=use_png
    )
    with open(os.path.join(output_dir, f"{base_name}_train.json"), "w") as f:
        json.dump(coco_json_train, f)

    coco_json_val = generate_coco_json(
        zod_frames, split="val", classes=classes, anonymization=anonymization, use_png=use_png
    )
    with open(os.path.join(output_dir, f"{base_name}_val.json"), "w") as f:
        json.dump(coco_json_val, f)

    typer.echo("Successfully converted ZOD to COCO format. Output files:")
    typer.echo(f"    train:  {output_dir}/{base_name}_train.json")
    typer.echo(f"    val:    {output_dir}/{base_name}_val.json")


if __name__ == "__main__":
    typer.run(convert_to_coco)

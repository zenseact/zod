"""Script that prepares the dataset for the traffic sign classification task.

Will create a dataset from the original zod frames by cropping out the relevant traffic signs
in each image and storing them for later use.
"""

import argparse
import json
import os
from dataclasses import dataclass
from itertools import repeat
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
from tqdm.contrib.concurrent import process_map

from zod.frames import annotation_parser as ap
from zod.frames.zod_frames import ZodFrames
from zod import constants
from zod.utils.zod_dataclasses import FrameInformation

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
class Args:
    """Script args."""

    dataset_root: str
    output_folder: str
    num_workers: Optional[int]
    padding_factor: Optional[float]
    padding_px: Optional[Tuple[int, int]]
    overwrite: bool


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-folder",
        type=str,
        required=True,
        help="Where the dataset should be saved.",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Path to the root of the dataset to use.",
    )
    parser.add_argument("--num-workers", type=int, help="Number of workers to use.")
    parser.add_argument(
        "--padding-factor",
        type=float,
        help="Factor to multiply the padding with.",
    )
    parser.add_argument(
        "--padding-px",
        type=int,
        nargs="+",
        help="Padding in x and y direction.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing files.",
    )

    args = parser.parse_args()

    assert not (
        args.padding_factor is not None and args.padding_px is not None
    ), "Cannot specify both padding and padding_factor"

    return Args(**vars(args))


def _process_frame(
    frame: FrameInformation, args: Args, train_ids: Set[str]
) -> List[Dict[str, Any]]:
    """Process a single frame."""
    if frame.traffic_sign_annotation_path is None:
        return []

    traffic_signs = ap.parse_traffic_sign_annotation(frame.traffic_sign_annotation_path)
    if len(traffic_signs) == 0:
        return []

    new_cropped_frames = []
    # load the image
    # TODO: change to blur image path before release
    image_path = frame.camera_frame[constants.CAMERA_FRONT_BLUR].filepath.replace(
        "blur", "original"
    )
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    for traffic_sign in traffic_signs:

        cls_name = traffic_sign.traffic_sign_class
        train_or_val = constants.TRAIN if frame.frame_id in train_ids else constants.VAL
        cls_folder = os.path.join(args.output_folder, train_or_val, cls_name)

        if not os.path.exists(cls_folder):
            os.makedirs(cls_folder)
            os.chmod(cls_folder, 0o777)

        new_frame_id = f"{frame.frame_id}_{traffic_sign.uuid}"
        output_file = os.path.join(
            cls_folder,
            f"{new_frame_id}.png",
        )
        if not args.overwrite and os.path.exists(output_file):
            continue

        # crop out the correct patch
        cropped_image, padding = traffic_sign.bounding_box.crop_from_image(
            image,
            padding_factor=args.padding_factor,
            padding=args.padding_px,
        )
        # save the image
        cv2.imwrite(output_file, cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))

        original_widht = traffic_sign.bounding_box.dimension[0]
        original_height = traffic_sign.bounding_box.dimension[1]
        center_x = padding[0] + original_widht // 2
        center_y = padding[1] + original_height // 2

        # create a frame info file
        new_cropped_frames.append(
            {
                "frame_id": new_frame_id,
                "image_path": os.path.join(cls_folder, f"{new_frame_id}.png"),
                "padding_left": padding[0],
                "padding_top": padding[1],
                "padding_right": padding[2],
                "padding_bottom": padding[3],
                "padded_width": cropped_image.shape[1],
                "padded_height": cropped_image.shape[0],
                "center_x": center_x,
                "center_y": center_y,
                "original_width": traffic_sign.bounding_box.dimension[0],
                "original_height": traffic_sign.bounding_box.dimension[1],
                "annotation": {
                    key: val for key, val in traffic_sign.__dict__.items() if key != "bounding_box"
                },
            }
        )

    return new_cropped_frames


def main(args: Args):
    """Run the main script."""

    print("Will create dataset from", args.dataset_root)
    print("Will save dataset to", args.output_folder)

    if not os.path.exists((args.output_folder)):
        os.makedirs(args.output_folder)
        os.chmod(args.output_folder, 0o777)
    zod_frames = ZodFrames(args.dataset_root, version="full")
    print(f"Will process {len(zod_frames)} full frames.")

    traffic_sign_frames = process_map(
        _process_frame,
        zod_frames.get_all_frames().values(),
        repeat(args),
        desc="Processing frames in ZOD",
        max_workers=args.num_workers,
        chunksize=1 if args.num_workers == 1 else 100,
    )

    # flatten the returned list
    traffic_sign_frames: List[Dict[str, Any]] = [
        frame for frames in traffic_sign_frames for frame in frames
    ]

    train_frame_ids = set(zod_frames.get_split(constants.TRAIN))
    val_frame_ids = set(zod_frames.get_split(constants.VAL))

    train_frames = [
        f for f in traffic_sign_frames if f["frame_id"].split("_")[0] in train_frame_ids
    ]
    val_frames = [f for f in traffic_sign_frames if f["frame_id"].split("_")[0] in val_frame_ids]

    # write it to a json file
    with open(os.path.join(args.output_folder, "dataset.json"), "w") as f:
        json.dump({constants.TRAIN: train_frames, constants.VAL: val_frames}, f)

    print(
        SUMMARY.format(
            len(traffic_sign_frames),
            args.output_folder,
            args.padding_factor if args.padding_factor is not None else args.padding_px,
        )
    )


if __name__ == "__main__":
    main(_parse_args())

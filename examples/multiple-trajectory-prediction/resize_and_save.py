"""Script to resize images and save them (robust to corrupt/truncated files)."""

import glob
import os
from pathlib import Path

from dataset.zod_configs import ZodConfigs
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


def looks_like_jpeg(path: str) -> bool:
    """Quick magic-byte check for JPEG."""
    try:
        with open(path, "rb") as f:
            return f.read(2) == b"\xff\xd8"
    except Exception:
        return False


def resize_images(source_directory: str, size: int) -> None:
    """Resize script to save a new ZOD single frames dataset to specified size.

    Data is saved as original_name_resized.jpg.
    """
    pattern = os.path.join(source_directory, "single_frames", "*", "camera_front_dnat", "*.jpg")
    files = glob.glob(pattern)

    bad_list_path = Path(source_directory) / "bad_images.txt"
    skipped = 0
    written = 0
    checked = 0

    with bad_list_path.open("w") as badlog:
        for file in tqdm(files, desc="Resizing images"):
            checked += 1

            # Skip already resized files
            if file.endswith("_resized.jpg"):
                continue

            if not looks_like_jpeg(file):
                badlog.write(f"not_jpeg_magic:{file}\n")
                skipped += 1
                continue

            # Try to open/verify and then reopen for actual load (verify invalidates the fp)
            try:
                with Image.open(file) as probe:
                    probe.verify()  # quick structural check

                with Image.open(file) as img:
                    img_rgb = img.convert("RGB")  # ensure JPEG-compatible
                    img_resized = img_rgb.resize((size, size), Image.Resampling.LANCZOS)

                destination_file = file.rsplit(".", 1)[0] + "_resized.jpg"
                img_resized.save(destination_file, "JPEG", quality=95, optimize=True)
                written += 1

            except (UnidentifiedImageError, OSError) as e:
                # OSError can happen on truncated/invalid images
                badlog.write(f"unreadable:{file}  reason:{type(e).__name__}: {e}\n")
                skipped += 1
                continue

    print(
        f"Processed {checked} files. Wrote {written} resized images. "
        f"Skipped {skipped} bad/unreadable files.\n"
        f"See bad image list at: {bad_list_path}"
    )


if __name__ == "__main__":
    configs = ZodConfigs()
    directory = configs.DATASET_ROOT
    size = configs.IMG_SIZE
    resize_images(directory, size)

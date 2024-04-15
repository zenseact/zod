"""Helper script to download the dataset using academic torrents."""

import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import List

from tqdm import tqdm

from zod.cli.dropbox_download import DownloadSettings, FilterSettings
from zod.cli.utils import SubDataset, Version
from zod.constants import MINI

TORRENT_WEBSITE = "https://academictorrents.com/"

ZOD_FRAMES_TORRENT = "329ae74426e067ef06b82241b5906400ca3caf03"
ZOD_SEQUENCES_TORRENT = "95ece9c22470c4f7df51a82bcc3407cc038419be"
ZOD_DRIVES_TORRENT = "a05ab2e524dc38c1498fcc9fb621329cc97e3837"

TORRENT_FILEMAP = {
    SubDataset.FRAMES: f"{TORRENT_WEBSITE}/download/{ZOD_FRAMES_TORRENT}.torrent",
    SubDataset.SEQUENCES: f"{TORRENT_WEBSITE}/download/{ZOD_SEQUENCES_TORRENT}.torrent",
    SubDataset.DRIVES: f"{TORRENT_WEBSITE}/download/{ZOD_DRIVES_TORRENT}.torrent",
}

TORRENT_WEBSITEMAP = {
    SubDataset.FRAMES: f"{TORRENT_WEBSITE}/details/{ZOD_FRAMES_TORRENT}",
    SubDataset.SEQUENCES: f"{TORRENT_WEBSITE}/details/{ZOD_SEQUENCES_TORRENT}",
    SubDataset.DRIVES: f"{TORRENT_WEBSITE}/details/{ZOD_DRIVES_TORRENT}",
}

SIZE_FILEMAP = {
    SubDataset.FRAMES: 9096553,
    SubDataset.SEQUENCES: 1280237,
    SubDataset.DRIVES: 290441,
}

DRIVES_FILES_TO_IDX = {
    "annotations": 1,
    "drives_mini": 2,
    "images_front_blur": 3,
    "infos": 4,
    "lidar_velodyne": 5,
    "oxts": 6,
    "vehicle_data": 7,
}

SEQUENCES_FILES_TO_IDX = {
    "annotations": 1,
    "images_front_blur": 2,
    "infos": 3,
    "lidar_velodyne": 4,
    "oxts": 5,
    "sequences_mini": 6,
    "vehicle_data": 7,
}

FRAMES_FILES_TO_IDX = {
    "annotations": 1,
    "frames_mini": 2,
    "images_front_blur": 3,
    "images_front_dnat": 4,
    "infos": 5,
    "lidar_velodyne_01after": 6,
    "lidar_velodyne_01before": 7,
    "lidar_velodyne_02after": 8,
    "lidar_velodyne_02before": 9,
    "lidar_velodyne_03after": 10,
    "lidar_velodyne_03before": 11,
    "lidar_velodyne_04after": 12,
    "lidar_velodyne_04before": 13,
    "lidar_velodyne_05after": 14,
    "lidar_velodyne_05before": 15,
    "lidar_velodyne_06after": 16,
    "lidar_velodyne_06before": 17,
    "lidar_velodyne_07after": 18,
    "lidar_velodyne_07before": 19,
    "lidar_velodyne_08after": 20,
    "lidar_velodyne_08before": 21,
    "lidar_velodyne_09after": 22,
    "lidar_velodyne_09before": 23,
    "lidar_velodyne_10after": 24,
    "lidar_velodyne_10before": 25,
    "lidar_velodyne": 26,
    "oxts": 27,
}


def check_aria_install(subset: SubDataset):
    # assert that aria2c is installed
    try:
        subprocess.check_call(["aria2c", "--version"])
    except FileNotFoundError:
        print(
            "aria2c is not installed. Please install aria2c to download the dataset. Using: `apt install aria2`"  # noqa
        )
        print(f"Alternatively, you can download the dataset manually from: ")
        print(f"\t {TORRENT_WEBSITEMAP[subset]}")
        sys.exit(1)


def get_files_to_download(subset: SubDataset, filter_settings: FilterSettings) -> List[int]:
    if subset == SubDataset.DRIVES:
        files_to_idx = DRIVES_FILES_TO_IDX
    elif subset == SubDataset.SEQUENCES:
        files_to_idx = SEQUENCES_FILES_TO_IDX
    elif subset == SubDataset.FRAMES:
        files_to_idx = FRAMES_FILES_TO_IDX
    else:
        raise ValueError(f"Unknown subset: {subset}")

    # mini dataset
    if filter_settings.version == Version.MINI:
        return [f"{subset.name.lower()}_mini.tar.gz"], [files_to_idx[f"{subset.name.lower()}_mini"]]

    files = []
    # annotations
    if filter_settings.annotations:
        files.append("annotations")

    # camera data
    if filter_settings.images:
        if filter_settings.blur:
            files.append("images_front_blur")
        # we only have dnat images for frames
        if filter_settings.dnat and subset == SubDataset.FRAMES:
            files.append("images_front_dnat")

    # gnss data
    if filter_settings.oxts:
        files.append("oxts")

    # lidar_velodyne data
    if filter_settings.lidar:
        files.append("lidar_velodyne")
        if subset == SubDataset.FRAMES:
            if not filter_settings.num_scans_before == 0:
                n_scans = (
                    10
                    if filter_settings.num_scans_before == -1
                    else filter_settings.num_scans_before
                )
                for i in range(1, n_scans + 1):
                    files.append(f"lidar_velodyne_{i:02d}before")

            if not filter_settings.num_scans_after == 0:
                n_scans = (
                    10 if filter_settings.num_scans_after == -1 else filter_settings.num_scans_after
                )
                for i in range(1, n_scans + 1):
                    files.append(f"lidar_velodyne_{i:02d}after")

    # vehicle data
    if filter_settings.vehicle_data:
        if subset != SubDataset.FRAMES:
            files.append("vehicle_data")

    # frame infos
    if filter_settings.infos:
        files.append("infos")

    file_indicies = [files_to_idx[f] for f in files]

    files = [f"{file}.tar.gz" for file in files]

    return files, file_indicies


def download_torrent_file(subset: SubDataset, output_dir: Path):
    """Download the torrent file for the given subset and version."""
    url = TORRENT_FILEMAP[subset]
    filename = f"{subset.name.lower()}.torrent"
    print("Downloading torrent file: {}".format(filename))
    try:
        urllib.request.urlretrieve(url, filename)
        # move the torrent file to the output directory
        os.rename(filename, output_dir / filename)

        assert os.path.getsize(output_dir / filename) == SIZE_FILEMAP[subset], (
            f"Downloaded torrent file is not the correct size. "
            f"Expected: {SIZE_FILEMAP[subset]}, "
            f"Actual: {os.path.getsize(output_dir / filename)}"
        )
        return output_dir / filename
    except Exception as e:
        print(f"Failed to download torrent file: {e}")
        print("Please ensure that you have the latest stable version of the development kit.")
        print("If the problem persists, please start a new issue on the GitHub repository.")
        exit(1)


def download_dataset(
    dl_settings: DownloadSettings, filter_settings: FilterSettings, subset: SubDataset
):
    # we have to make sure aria2c is installed
    if not dl_settings.dry_run:
        check_aria_install(subset)

    dl_dir = Path(dl_settings.output_dir) / "downloads"
    dl_dir.mkdir(parents=True, exist_ok=True)

    torrent_filepath = download_torrent_file(subset, dl_dir)

    files, file_indicies = get_files_to_download(subset, filter_settings)
    # create the aria2c command
    # seed-time sets the number of seconds to seed the torrent after downloading
    # continue will continue the download if the torrent has already been partially downloaded
    cmd = [
        "aria2c",
        f"--select-file={','.join([str(i) for i in file_indicies])}",
        f"--torrent-file={torrent_filepath}",
        f"--dir={str(dl_dir)}",
        "--seed-time=0",
        "--continue=true",
    ]

    if dl_settings.dry_run:
        print(f"Would download: {files}")
        print("Would run: {}".format(" ".join(cmd)))
        return

    try:
        subprocess.check_call(cmd)
    except Exception as e:
        print("Failed to download the dataset. Error: {}".format(e))
        print("Please ensure that you have the latest stable version of the development kit.")
        print("If the problem persists, please start a new issue on the GitHub repository.")
        exit(1)

    # aria2c will download adjacent files in the torrent, so we need to remove those files
    # that we don't want
    dl_dir = dl_dir / subset.name.lower()
    for f in dl_dir.iterdir():
        if f.name not in files:
            f.unlink()

    # if we are extracting, extract the tar gz files
    if dl_settings.extract:
        for f in dl_dir.iterdir():
            if f.name.endswith(".tar.gz"):
                with tarfile.open(f, "r:gz") as tar:
                    for member in tqdm(
                        iterable=tar.getmembers(),
                        total=len(tar.getmembers()),
                        desc=f"Extracting {f.name}",
                    ):
                        tar.extract(member=member, path=dl_settings.output_dir)

    # if we are removing the archives, remove them
    if dl_settings.rm:
        shutil.rmtree(dl_dir)

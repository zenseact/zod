"""Helper script to download the dataset using academic torrents."""

import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import List

import typer
from tqdm import tqdm

from zod.cli.download import DownloadSettings, FilterSettings, _print_summary
from zod.cli.utils import SubDataset, Version
from zod.constants import MINI

TORRENT_FILEMAP = {
    SubDataset.FRAMES: "https://academictorrents.com/download/329ae74426e067ef06b82241b5906400ca3caf03.torrent",  # noqa
    SubDataset.SEQUENCES: "https://academictorrents.com/download/95ece9c22470c4f7df51a82bcc3407cc038419be.torrent",  # noqa
    SubDataset.DRIVES: "https://academictorrents.com/download/a05ab2e524dc38c1498fcc9fb621329cc97e3837.torrent",  # noqa
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

app = typer.Typer(help="Zenseact Open Dataset Downloader", no_args_is_help=True)


def check_aria_install():
    # assert that aria2c is installed
    try:
        subprocess.check_call(["aria2c", "--version"])
    except FileNotFoundError:
        print(
            "aria2c is not installed. Please install aria2c to download the dataset. Using: `apt install aria2`"  # noqa
        )
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
        exit(1)


def _download_dataset(
    dl_settings: DownloadSettings, filter_settings: FilterSettings, subset: SubDataset
):
    dl_dir = Path(dl_settings.output_dir) / "downloads"
    dl_dir.mkdir(parents=True, exist_ok=True)

    torrent_filepath = download_torrent_file(subset, dl_dir)

    files, file_indicies = get_files_to_download(subset, filter_settings)
    # create the aria2c command
    cmd = [
        "aria2c",
        f"--select-file={','.join([str(i) for i in file_indicies])}",
        f"--torrent-file={torrent_filepath}",
        f"--dir={str(dl_dir)}",
        "--seed-time=0",
        "--continue=true",
    ]

    if dl_settings.dry_run:
        print(cmd)
        cmd.append("--dry-run")

    subprocess.check_call(cmd)

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


REQ = "Required Arguments"
GEN = "General Download Settings"
FIL = "General Filter Settings"
FRA = "Frames Filter Settings"
SEQ = "Sequences/Drives Filter Settings"


@app.command()
def download(
    output_dir: str = typer.Option(
        ...,
        help="Output directory where dataset will be extracted",
        prompt="Where do you want to extract the dataset (e.g. ~/data/zod)?",
        prompt_required=False,
        rich_help_panel=REQ,
    ),
    subset: SubDataset = typer.Option(
        ...,
        help="The sub-dataset to download",
        prompt="Which sub-dataset do you want to download?",
        prompt_required=False,
        rich_help_panel=REQ,
    ),
    version: Version = typer.Option(
        ...,
        help="The version of the dataset to download",
        prompt="Which version do you want to download?",
        prompt_required=False,
        rich_help_panel=REQ,
    ),
    # General download settings
    rm: bool = typer.Option(False, help="Remove the downloaded archives", rich_help_panel=GEN),
    dry_run: bool = typer.Option(False, help="Print what would be downloaded", rich_help_panel=GEN),
    extract: bool = typer.Option(True, help="Unpack the archives", rich_help_panel=GEN),
    extract_already_downloaded: bool = typer.Option(
        False, help="Extract already downloaded archives", rich_help_panel=GEN
    ),
    no_confirm: bool = typer.Option(
        False,
        "-y",
        "--no-confirm/--confirm",
        help="Don't ask for confirmation",
        is_flag=True,
        flag_value=False,
        rich_help_panel=GEN,
    ),
    # Filter settings
    annotations: bool = typer.Option(True, help="Download annotations", rich_help_panel=FIL),
    images: bool = typer.Option(True, help="Whether to download the images", rich_help_panel=FIL),
    blur: bool = typer.Option(True, help="Download blur images", rich_help_panel=FIL),
    lidar: bool = typer.Option(True, help="Download lidar data", rich_help_panel=FIL),
    oxts: bool = typer.Option(True, help="Download oxts data", rich_help_panel=FIL),
    infos: bool = typer.Option(True, help="Download infos", rich_help_panel=FIL),
    vehicle_data: bool = typer.Option(True, help="Download the vehicle data", rich_help_panel=SEQ),
    dnat: bool = typer.Option(False, help="Download DNAT images", rich_help_panel=FRA),
    num_scans_before: int = typer.Option(
        0,
        help="Number of earlier lidar scans to download (-1 == all)",
        rich_help_panel=FRA,
    ),
    num_scans_after: int = typer.Option(
        0, help="Number of later lidar scans to download (-1 == all)", rich_help_panel=FRA
    ),
    parallel: bool = typer.Option(
        False,
        help="Not used in this script. Exist for compatibility with zod/cli/download.py",
        rich_help_panel=GEN,
    ),
    max_workers: int = typer.Option(
        1,
        help="Not used in this script. Exist for compatibility with zod/cli/download.py",
        rich_help_panel=GEN,
    ),
):
    # we have to make sure aria2c is installed
    check_aria_install()

    # initialize the download settings
    download_settings = DownloadSettings(
        url="",
        output_dir=os.path.expanduser(output_dir),
        rm=rm,
        dry_run=dry_run,
        extract=extract,
        extract_already_downloaded=extract_already_downloaded,
        parallel=False,
        max_workers=1,
    )

    # initialize the filter settings
    filter_settings = FilterSettings(
        version=version,
        annotations=annotations,
        images=images,
        blur=blur,
        dnat=dnat,
        lidar=lidar,
        oxts=oxts,
        infos=infos,
        vehicle_data=vehicle_data,
        num_scans_before=num_scans_before if subset == SubDataset.FRAMES else -1,
        num_scans_after=num_scans_after if subset == SubDataset.FRAMES else -1,
    )

    if not no_confirm:
        _print_summary(download_settings, filter_settings, subset)
        typer.confirm(
            f"Download with the above settings?",
            abort=True,
        )

    _download_dataset(download_settings, filter_settings, subset)


if __name__ == "__main__":
    app()

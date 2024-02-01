"""This script is to be used to download the Zenseact Open Dataset."""
import os
from enum import Enum

from zod.cli.dropbox_download import DownloadSettings, FilterSettings
from zod.cli.dropbox_download import download_dataset as drobbox_download
from zod.cli.torrent_download import download_dataset as torrent_download
from zod.cli.utils import SubDataset, Version

try:
    import typer
except ImportError:
    print('zod is installed without the CLI dependencies: pip install "zod[cli]"')
    exit(1)


app = typer.Typer(help="Zenseact Open Dataset Downloader", no_args_is_help=True)


class DownloadSource(str, Enum):
    DROPBOX = "dropbox"
    TORRENT = "torrent"


def _print_summary(download_settings, filter_settings, subset):
    print("\nDownload settings:")
    print(download_settings)
    print("\nFilter settings:")
    if filter_settings.version == Version.MINI:
        print("    version: mini\n    (other settings are ignored for mini)")
    else:
        print(filter_settings)
        if subset == SubDataset.FRAMES and (
            filter_settings.num_scans_before == filter_settings.num_scans_after == 0
        ):
            typer.secho(
                "Note! The current settings will only download the core lidar frames. "
                "If you need surrounding scans, set --num-scans-before and/or --num-scans-after.",
                fg=typer.colors.YELLOW,
            )
    print("\n")


REQ = "Required Arguments"
GEN = "General Download Settings"
FIL = "General Filter Settings"
FRA = "Frames Filter Settings"
SEQ = "Sequences/Drives Filter Settings"


@app.command()
def download(
    url: str = typer.Option(
        ...,
        help="The dropbox shared folder url",
        prompt="What is the dropbox url to the dataset (you can get it from zod.zenseact.com)? Enter anything if you are downloading using torrent.",  # noqa
        prompt_required=False,
        rich_help_panel=REQ,
    ),
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
    source: DownloadSource = typer.Option(
        ...,
        "--source",
        "-s",
        help="The source of the dataset",
        prompt="Where do you want to download the dataset from? (torrent or dropbox)",
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
    parallel: bool = typer.Option(True, help="Download files in parallel", rich_help_panel=GEN),
    max_workers: int = typer.Option(
        None, help="Max number of workers for parallel downloads", rich_help_panel=GEN
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
        0, help="Number of earlier lidar scans to download (-1 == all)", rich_help_panel=FRA
    ),
    num_scans_after: int = typer.Option(
        0, help="Number of later lidar scans to download (-1 == all)", rich_help_panel=FRA
    ),
):
    """Download the Zenseact Open Dataset."""
    download_settings = DownloadSettings(
        url=url,
        output_dir=os.path.expanduser(output_dir),
        rm=rm,
        dry_run=dry_run,
        extract=extract,
        extract_already_downloaded=extract_already_downloaded,
        parallel=parallel,
        max_workers=max_workers,
    )
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

    if source == DownloadSource.DROPBOX:
        drobbox_download(download_settings, filter_settings, subset.folder)
    elif source == DownloadSource.TORRENT:
        torrent_download(download_settings, filter_settings, subset)


if __name__ == "__main__":
    app()

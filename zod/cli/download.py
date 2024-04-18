"""This script is to be used to download the Zenseact Open Dataset."""

import contextlib
import os
import os.path as osp
import subprocess
import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import requests
from tqdm import tqdm

from zod.cli.utils import SubDataset, Version

try:
    import click.exceptions
    import dropbox
    import dropbox.exceptions
    import dropbox.files
    import dropbox.sharing
    import typer
except ImportError:
    print('zod is installed without the CLI dependencies: pip install "zod[cli]"')
    exit(1)


APP_KEY = "kcvokw7c7votepo"
REFRESH_TOKEN = "z2h_5rjphJEAAAAAAAAAAUWtQlltCYlMbZ2WqMgtymELhiSuKIksxAYeArubKnZV"
CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB
TIMEOUT = 8 * 60 * 60  # 8 hours

app = typer.Typer(help="Zenseact Open Dataset Downloader", no_args_is_help=True)


@dataclass
class DownloadExtractInfo:
    """Information about a file to extract."""

    url: str
    file_path: str
    extract_dir: str
    dl_dir: str
    rm: bool
    dry_run: bool
    size: int
    content_hash: str
    extract: bool
    extract_already_downloaded: bool


@dataclass
class FilterSettings:
    """Filter settings."""

    version: Version
    annotations: bool
    images: bool
    blur: bool
    dnat: bool
    lidar: bool
    oxts: bool
    infos: bool
    vehicle_data: bool
    num_scans_before: int
    num_scans_after: int

    def __str__(self):
        return "\n".join([f"    {key}: {value}" for key, value in self.__dict__.items()])


@dataclass
class DownloadSettings:
    """Download settings."""

    url: str
    output_dir: str
    rm: bool
    dry_run: bool
    extract: bool
    extract_already_downloaded: bool
    parallel: bool
    max_workers: int = 8

    def __str__(self):
        return "\n".join([f"    {key}: {value}" for key, value in self.__dict__.items()])


class ResumableDropbox(dropbox.Dropbox):
    """A patched dropbox client that allows to resume downloads."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._headers = {} if self._headers is None else self._headers
        self._lock = threading.Lock()

    def sharing_get_shared_link_file(self, url, path=None, link_password=None, start=0):
        """
        Download the shared link's file from a user's Dropbox.
        """
        # Need to lock here because the headers are shared between all requests
        with self._lock:
            self._headers["Range"] = f"bytes={start}-"
            res = super().sharing_get_shared_link_file(url, path=path, link_password=link_password)
            del self._headers["Range"]
        return res


def _print_final_msg(pbar: tqdm, basename: str):
    """Print a final message for each downloaded file, with stats and nice padding."""
    msg = "Downloaded " + basename + " " * (45 - len(basename))
    total_time = pbar.format_dict["elapsed"]
    size_in_mb = pbar.n / 1024 / 1024
    speed_in_mb = size_in_mb / total_time
    for name, val, unit in [
        ("time", total_time, "s"),
        ("size", size_in_mb, "MB"),
        ("speed", speed_in_mb, "MB/s"),
    ]:
        val = f"{val:.2f}{unit}"
        msg += f"{name}: {val}" + " " * (14 - len(val))
    tqdm.write(msg)


def _download(download_path: str, dbx: ResumableDropbox, info: DownloadExtractInfo):
    current_size = 0
    if osp.exists(download_path):
        current_size = osp.getsize(download_path)
    basename = osp.basename(info.file_path)
    pbar = tqdm(
        total=info.size,
        unit="iB",
        unit_scale=True,
        desc=f"Downloading {basename}...",
        leave=False,
        initial=current_size,
    )
    if pbar.n > info.size:
        tqdm.write(
            f"Error! File {download_path} already exists and is larger than expected. " "Please delete and try again."
        )
    if pbar.n > 0:
        # this means we are retrying or resuming a previously interrupted download
        tqdm.write(f"Resuming download of {download_path} from {current_size} bytes.")
    # Retry download if partial file exists (e.g. due to network error)
    while pbar.n < info.size:
        try:
            _, response = dbx.sharing_get_shared_link_file(url=info.url, path=info.file_path, start=pbar.n)
            with open(download_path, "ab") as f:
                with contextlib.closing(response):
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:  # filter out keep-alive new chunks
                            size = f.write(chunk)
                            pbar.update(size)
        except requests.exceptions.ChunkedEncodingError:
            continue  # This can happen when dropbox unexpectly closes the connection
    # Final checks and printouts
    assert osp.getsize(download_path) == info.size
    _print_final_msg(pbar, basename)


def _extract(tar_path: str, output_dir: str):
    """Extract a tar file to a directory."""
    pbar = tqdm(desc=f"Extracting {osp.basename(tar_path)}...", leave=False)
    # Check if system has tar installed (pipe stderr to None to avoid printing)
    if os.system("tar --version > /dev/null 2>&1") == 0:
        with subprocess.Popen(
            ["tar", "-xvf", tar_path, "-C", output_dir],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ) as tar:
            for _ in tar.stdout:
                pbar.update(1)
        # check if tar exited with error
        if tar.returncode != 0:
            tqdm.write(f"Error extracting {tar_path} with tar.")
    else:
        # Fallback to python tarfile
        with tarfile.open(name=tar_path) as tar:
            # pbar.total = len(tar.getmembers())
            for member in tar.getmembers():
                tar.extract(member=member, path=output_dir)
                pbar.update(1)
    tqdm.write(f"Extracted {pbar.n} files from {osp.basename(tar_path)}.")


def _already_downloaded(download_path: str, target_size: int):
    """Check if the file is already downloaded."""
    old_format_path = download_path.rsplit("_", 1)[0]  # /path/to/file_abc123 -> /path/to/file
    if osp.exists(download_path) and osp.getsize(download_path) == target_size:
        return True
    elif osp.exists(old_format_path) and osp.getsize(old_format_path) == target_size:
        return True
    return False


def _download_and_extract(dbx: ResumableDropbox, info: DownloadExtractInfo):
    """Download a file from a Dropbox share link to a local path."""
    should_extract = info.extract
    file_name = osp.basename(info.file_path)
    data_name = file_name.split(".")[0]
    download_path = osp.join(info.dl_dir, f"{file_name}_{info.content_hash[:8]}")
    if _already_downloaded(download_path, info.size):
        if not info.extract_already_downloaded:
            tqdm.write(f"{data_name} already exists. Skipping download and extraction.")
            should_extract = False
        else:
            tqdm.write(f"{data_name} already exists. Skipping download.")
    elif info.dry_run:
        msg = "download" if not should_extract else "download and extract"
        typer.echo(f"Would {msg} {info.file_path} to {download_path}")
        return
    else:
        try:
            _download(download_path, dbx, info)
        except Exception as e:
            print(f"Error downloading {data_name}: {e}. Please retry")
            return

    if should_extract:
        try:
            _extract(download_path, info.extract_dir)
        except Exception as e:
            print(f"Error extracting {data_name}: {e}. Please retry")
            return

    if info.rm and not info.dry_run:
        os.remove(download_path)


def _filter_entry(entry: dropbox.files.Metadata, settings: FilterSettings) -> bool:
    """Filter the entry based on the flags."""
    if settings.version == Version.MINI:
        return "mini" in entry.name
    elif "mini" in entry.name:
        return False

    if not settings.annotations and "annotations" in entry.name:
        return False
    if "images" in entry.name:
        if not settings.images:
            return False
        if not settings.blur and "blur" in entry.name:
            return False
        if not settings.dnat and "dnat" in entry.name:
            return False
    if not settings.oxts and "oxts" in entry.name:
        return False
    if not settings.infos and "infos" in entry.name:
        return False
    if not settings.vehicle_data and "vehicle_data" in entry.name:
        return False
    if "lidar" in entry.name:
        if not settings.lidar:
            return False
        if "after" in entry.name or "before" in entry.name:
            # this is only the case for frames, where we have surrounding frames
            distance = int(entry.name.split("_")[2][:2])
            tgt = settings.num_scans_before if "before" in entry.name else settings.num_scans_after
            if tgt != -1 and distance > tgt:
                return False
    return True


def _download_dataset(dl_settings: DownloadSettings, filter_settings: FilterSettings, dirname: str):
    dbx = ResumableDropbox(app_key=APP_KEY, oauth2_refresh_token=REFRESH_TOKEN, timeout=TIMEOUT)
    entries = _list_folder(dl_settings.url, dbx, dirname)
    if not entries:
        typer.echo(
            "Warning! No files found. Check the url, but this could be a known dropbox error.\n"
            "We are working on a fix, so please try again layer. Sorry for the inconvenience."
        )
        return
    dl_dir = osp.join(dl_settings.output_dir, "downloads", dirname)
    files_to_download = [
        DownloadExtractInfo(
            url=dl_settings.url,
            file_path=f"/{dirname}/" + entry.name,
            dl_dir=dl_dir,
            extract_dir=dl_settings.output_dir,
            size=entry.size,
            content_hash=entry.content_hash,
            rm=dl_settings.rm,
            dry_run=dl_settings.dry_run,
            extract=dl_settings.extract,
            extract_already_downloaded=dl_settings.extract_already_downloaded,
        )
        for entry in entries
        if _filter_entry(entry, filter_settings)
    ]
    if not files_to_download:
        typer.echo("No files to download. Perhaps you specified too many filters?")
        return

    if not dl_settings.dry_run:
        os.makedirs(dl_settings.output_dir, exist_ok=True)
        os.makedirs(dl_dir, exist_ok=True)
    if dl_settings.parallel:
        with ThreadPoolExecutor(max_workers=dl_settings.max_workers) as pool:
            pool.map(lambda info: _download_and_extract(dbx, info), files_to_download)
    else:
        for info in files_to_download:
            _download_and_extract(dbx, info)


def _list_folder(url, dbx: ResumableDropbox, path: str):
    shared_link = dropbox.files.SharedLink(url=url)
    try:
        res = dbx.files_list_folder(path=f"/{path}", shared_link=shared_link)
    except dropbox.exceptions.ApiError as err:
        raise click.exceptions.ClickException(
            f"Dropbox raised the following error:\n\t{err}\nThis could be due to:"
            '\n\ta) bad url. Please try it in a browser and specify with quotes (--url="<url>").'
            "\n\tb) zod bandwidth limit. Sorry about this, and please try again the next day."
            "\n\tc) other error (bad internet connection, dropbox outage, etc.)."
        )

    entries = res.entries
    while res.has_more:
        res = dbx.files_list_folder_continue(res.cursor)
        entries.extend(res.entries)
    return entries


def _print_summary(download_settings, filter_settings, subset):
    print("\nDownload settings:")
    print(download_settings)
    print("\nFilter settings:")
    if filter_settings.version == Version.MINI:
        print("    version: mini\n    (other settings are ignored for mini)")
    else:
        print(filter_settings)
        if subset == SubDataset.FRAMES and (filter_settings.num_scans_before == filter_settings.num_scans_after == 0):
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
        prompt="What is the dropbox url to the dataset (you can get it from zod.zenseact.com)?",
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
    # General download settings
    rm: bool = typer.Option(False, help="Remove the downloaded archives", rich_help_panel=GEN),
    dry_run: bool = typer.Option(False, help="Print what would be downloaded", rich_help_panel=GEN),
    extract: bool = typer.Option(True, help="Unpack the archives", rich_help_panel=GEN),
    extract_already_downloaded: bool = typer.Option(
        False, help="Extract already downloaded archives", rich_help_panel=GEN
    ),
    parallel: bool = typer.Option(True, help="Download files in parallel", rich_help_panel=GEN),
    max_workers: int = typer.Option(None, help="Max number of workers for parallel downloads", rich_help_panel=GEN),
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
    _download_dataset(download_settings, filter_settings, subset.folder)


if __name__ == "__main__":
    app()

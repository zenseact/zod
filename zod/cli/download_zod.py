"""This script is to be used to download the zenseact open dataset."""
import contextlib
import os
import os.path as osp
import subprocess
import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from tqdm import tqdm

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
CHUNK_SIZE = 1024 * 1024  # 1 MB
TIMEOUT = 60 * 60  # 1 hour

app = typer.Typer(help="Zenseact Open Dataset Donwloader", no_args_is_help=True)


@dataclass
class ExtractInfo:
    """Information about a file to extract."""

    url: str
    file_path: str
    output_dir: str
    rm: bool
    dry_run: bool
    size: int
    extract: bool


@dataclass
class FilterSettings:
    """Filter settings."""

    mini: bool
    annotations: bool
    images: bool
    blur: bool
    dnat: bool
    lidar: bool
    oxts: bool
    calibrations: bool
    num_scans_before: int
    num_scans_after: int


@dataclass
class DownloadSettings:
    """Download settings."""

    url: str
    output_dir: str
    rm: bool
    dry_run: bool
    extract: bool
    parallel: bool


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

    # Print various download stats
    total_time = pbar.format_dict["elapsed"]
    size_in_mb = pbar.n / 1024 / 1024
    speed = size_in_mb / total_time
    for name, val, unit in [
        ("time", total_time, "s"),
        ("size", size_in_mb, "MB"),
        ("speed", speed, "MB/s"),
    ]:
        val = f"{val:.2f}{unit}"
        msg += f"{name}: {val}" + " " * (14 - len(val))

    tqdm.write(msg)


def _download(download_path: str, dbx: ResumableDropbox, info: ExtractInfo):
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
    # Retry download if partial file exists (e.g. due to network error)
    while pbar.n < info.size:
        if pbar.n > 0:
            # this means we are retrying or resuming a previously interrupted download
            tqdm.write(f"Resuming download of {download_path} from {current_size} bytes.")
        _, response = dbx.sharing_get_shared_link_file(
            url=info.url, path=info.file_path, start=pbar.n
        )
        with open(download_path, "ab") as f:
            with contextlib.closing(response):
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:  # filter out keep-alive new chunks
                        size = f.write(chunk)
                        pbar.update(size)
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
            for line in tar.stdout:
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


def _download_and_extract(dbx: ResumableDropbox, info: ExtractInfo):
    """Download a file from a Dropbox share link to a local path."""
    download_path = osp.join(info.output_dir, "downloads", osp.basename(info.file_path))
    if osp.exists(download_path) and osp.getsize(download_path) == info.size:
        tqdm.write(f"File {download_path} already exists. Skipping download.")
    elif info.dry_run:
        typer.echo(f"Would download and extract {info.file_path} to {download_path}")
        return
    else:
        _download(download_path, dbx, info)

    if info.extract:
        if info.dry_run:
            typer.echo(f"Would extract {download_path} to {info.output_dir}")
            return
        else:
            _extract(download_path, info.output_dir)

    if info.rm and not info.dry_run:
        os.remove(download_path)


def _filter_entry(entry: dropbox.files.Metadata, settings: FilterSettings) -> bool:
    """Filter the entry based on the flags."""
    if settings.mini and "mini" not in entry.name:
        # If mini is set, we only want to download the mini dataset
        return False
    # If we are downloading the full dataset, we don't want the mini dataset
    if not settings.mini and "mini" in entry.name:
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
    if not settings.calibrations and "calibrations" in entry.name:
        return False
    if "lidar" in entry.name:
        if not settings.lidar:
            return False
        distance = entry.name.split("_")[2][:2]
        if "after" in entry.name and (int(distance) > settings.num_scans_after):
            return False
        if "before" in entry.name and (int(distance) > settings.num_scans_before):
            return False
    return True


def _download_dataset(dl_settings: DownloadSettings, filter_settings: FilterSettings, dirname: str):
    dbx = ResumableDropbox(app_key=APP_KEY, oauth2_refresh_token=REFRESH_TOKEN, timeout=TIMEOUT)
    url, entries = _list_folder(dl_settings.url, dbx, dirname)
    files_to_download = [
        ExtractInfo(
            file_path=f"/{dirname}/" + entry.name,
            size=entry.size,
            url=url,
            output_dir=dl_settings.output_dir,
            rm=dl_settings.rm,
            dry_run=dl_settings.dry_run,
            extract=dl_settings.extract,
        )
        for entry in entries
        if _filter_entry(entry, filter_settings)
    ]
    if not files_to_download:
        typer.echo("No files to download. Perhaps you specified too many filters?")
        return

    if not dl_settings.dry_run:
        os.makedirs(dl_settings.output_dir, exist_ok=True)
        os.makedirs(osp.join(dl_settings.output_dir, "downloads"), exist_ok=True)
    if dl_settings.parallel:
        with ThreadPoolExecutor() as pool:
            pool.map(lambda info: _download_and_extract(dbx, info), files_to_download)
    else:
        for info in files_to_download:
            _download_and_extract(dbx, info)


def _list_folder(url, dbx, path):
    shared_link = dropbox.files.SharedLink(url=url)
    try:
        res = dbx.files_list_folder(path=f"/{path}", shared_link=shared_link)
    except dropbox.exceptions.ApiError as err:
        raise click.exceptions.ClickException(
            f"Dropbox raised the following error:\n\t{err}\nThis could be due to a bad url."
        )

    entries = res.entries
    while res.has_more:
        res = dbx.files_list_folder_continue(res.cursor)
        entries.extend(res.entries)
    return url, entries


@app.callback(no_args_is_help=True)
def common(
    ctx: typer.Context,
    url: str = typer.Option(..., help="The dropbox shared folder url"),
    output_dir: str = typer.Option(..., help="The output directory"),
    rm: bool = typer.Option(False, help="Whether to remove the downloaded archives"),
    dry_run: bool = typer.Option(
        False, help="Whether to only print the files that would be downloaded"
    ),
    extract: bool = typer.Option(True, help="Whether to unpack the archives"),
    parallel: bool = typer.Option(True, help="Whether to download files in parallel"),
):
    ctx.obj = DownloadSettings(
        url=url,
        output_dir=output_dir,
        rm=rm,
        dry_run=dry_run,
        extract=extract,
        parallel=parallel,
    )


@app.command()
def frames(
    ctx: typer.Context,
    mini: bool = typer.Option(False, help="Whether to download the mini dataset"),
    annotations: bool = typer.Option(True, help="Whether to download the annotations"),
    images: bool = typer.Option(True, help="Whether to download the images"),
    blur: bool = typer.Option(True, help="Whether to download the blur images"),
    dnat: bool = typer.Option(False, help="Whether to download the dnat images"),
    lidar: bool = typer.Option(True, help="Whether to download the lidar data"),
    num_scans_before: int = typer.Option(
        0, help="Number of earlier lidar scans to download (-1 == all)"
    ),
    num_scans_after: int = typer.Option(
        0, help="Number of later lidar scans to download (-1 == all)"
    ),
    oxts: bool = typer.Option(True, help="Whether to download the oxts data"),
    calibrations: bool = typer.Option(True, help="Whether to download the calibration data"),
):
    """Download the zenseact open dataset."""

    if images:
        assert blur or dnat, "Must download at least one type of image"

    dl_settings: DownloadSettings = ctx.obj
    filter_settings = FilterSettings(
        mini=mini,
        annotations=annotations,
        images=images,
        blur=blur,
        dnat=dnat,
        lidar=lidar,
        oxts=oxts,
        calibrations=calibrations,
        num_scans_before=num_scans_before,
        num_scans_after=num_scans_after,
    )
    _download_dataset(dl_settings, filter_settings, "single_frames")


@app.command()
def sequences(
    ctx: typer.Context,
    mini: bool = typer.Option(False, help="Whether to download the mini dataset"),
    annotations: bool = typer.Option(True, help="Whether to download the annotations"),
    images: bool = typer.Option(True, help="Whether to download the images"),
    lidar: bool = typer.Option(True, help="Whether to download the lidar data"),
    oxts: bool = typer.Option(True, help="Whether to download the oxts data"),
    calibrations: bool = typer.Option(True, help="Whether to download the calibration data"),
):
    """Download the zenseact open dataset."""
    dl_settings: DownloadSettings = ctx.obj
    filter_settings = FilterSettings(
        mini=mini,
        annotations=annotations,
        images=images,
        blur=True,
        dnat=False,
        lidar=lidar,
        oxts=oxts,
        calibrations=calibrations,
        num_scans_before=-1,
        num_scans_after=-1,
    )
    _download_dataset(dl_settings, filter_settings, "sequences")


if __name__ == "__main__":
    app()

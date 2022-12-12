"""This script is to be used to download the zenseact open dataset."""
import os
import os.path as osp
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import typer
from tqdm import tqdm


try:
    import dropbox
    import dropbox.files
except:
    print("Please install dropbox: pip install dropbox")
    exit(1)


APP_KEY = "kcvokw7c7votepo"
REFRESH_TOKEN = "z2h_5rjphJEAAAAAAAAAAUWtQlltCYlMbZ2WqMgtymELhiSuKIksxAYeArubKnZV"
CHUNK_SIZE = 1024 * 1024  # 1 MB


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
    test: bool
    images: bool
    lidar: bool
    oxts: bool
    num_scans_before: int
    num_scans_after: int


def _print_final_msg(pbar: tqdm, total_size: float, basename: str):
    """Print a final message for each downloaded file, with stats and nice padding."""
    msg = "Downloaded " + basename + " " * (40 - len(basename))

    # Print various download stats
    total_time = pbar.format_dict["elapsed"]
    size_in_mb = total_size / 1024 / 1024
    speed = size_in_mb / total_time
    for name, val, unit in [
        ("time", total_time, "s"),
        ("size", size_in_mb, "MB"),
        ("speed", speed, "MB/s"),
    ]:
        val = f"{val:.2f}{unit}"
        msg += f"{name}: {val}" + " " * (10 - len(val))

    tqdm.write(msg)


def _download(download_path: str, dbx: dropbox.Dropbox, info: ExtractInfo):
    # Perform request
    _, response = dbx.sharing_get_shared_link_file(url=info.url, path=info.file_path)
    # Stream request content
    total_size = int(response.headers.get("content-length", 0))
    basename = osp.basename(info.file_path)
    pbar = tqdm(
        total=total_size, unit="iB", unit_scale=True, desc=f"Downloading {basename}...", leave=False
    )
    with open(download_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            size = f.write(chunk)
            pbar.update(size)
    # Final checks and printouts
    assert response.status_code == 200
    _print_final_msg(pbar, total_size, basename)


def _download_and_extract(dbx: dropbox.Dropbox, info: ExtractInfo):
    """Download a file from a Dropbox share link to a local path."""
    download_path = osp.join(info.output_dir, "downloads", osp.basename(info.file_path))
    if not osp.exists(download_path):
        if info.dry_run:
            typer.echo(f"Would download {info.file_path} to {download_path}")
            return
        else:
            _download(download_path, dbx, info)
    else:
        typer.echo(f"File {download_path} already exists. Skipping download.")

    if info.extract:
        shutil.unpack_archive(download_path, info.output_dir)

    if info.rm:
        os.remove(download_path)


def _filter_entry(entry: dropbox.files.Metadata, settings: FilterSettings) -> bool:
    """Filter the entry based on the flags."""
    if settings.mini and "mini" not in entry.name:
        return False
    if not settings.mini and "mini" in entry.name:
        return False
    if settings.test and "test" not in entry.name:
        return False
    if not settings.test and "test" in entry.name:
        return False
    if not settings.images and "images" in entry.name:
        return False
    if not settings.oxts and "oxts" in entry.name:
        return False
    if "lidar" in entry.name:
        if not settings.lidar:
            return False
        if "after" in entry.name and (
            int(entry.name.split("_")[2][:-5]) > settings.num_scans_after
        ):
            return False
        if "before" in entry.name and (
            int(entry.name.split("_")[2][:-5]) > settings.num_scans_before
        ):
            return False
    return True


def download_zod(
    url: str = typer.Option(..., help="The dropbox shared folder url"),
    output_dir: str = typer.Option(..., help="The output directory"),
    mini: bool = typer.Option(False, help="Whether to download the mini dataset"),
    test: bool = typer.Option(False, help="Whether to download the test files"),
    images: bool = typer.Option(True, help="Whether to download the images"),
    lidar: bool = typer.Option(True, help="Whether to download the lidar data"),
    num_scans_before: int = typer.Option(
        0, help="Number of earlier lidar scans to download (-1 == all)"
    ),
    num_scans_after: int = typer.Option(
        0, help="Number of later lidar scans to download (-1 == all)"
    ),
    oxts: bool = typer.Option(True, help="Whether to download the oxts data"),
    rm: bool = typer.Option(False, help="Whether to remove the downloaded archives"),
    dry_run: bool = typer.Option(
        False, help="Whether to only print the files that would be downloaded"
    ),
    extract: bool = typer.Option(True, help="Whether to unpack the archives"),
    parallel: bool = typer.Option(True, help="Whether to download files in parallel"),
):
    """Download the zenseact open dataset."""
    filter_settings = FilterSettings(
        mini=mini,
        test=test,
        images=images,
        lidar=lidar,
        oxts=oxts,
        num_scans_before=num_scans_before,
        num_scans_after=num_scans_after,
    )

    dbx = dropbox.Dropbox(app_key=APP_KEY, oauth2_refresh_token=REFRESH_TOKEN)
    url = url.replace("hdl", "h?dl")
    shared_link = dropbox.files.SharedLink(url=url)
    res = dbx.files_list_folder(path="/single_frames", shared_link=shared_link)
    if not dry_run:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(osp.join(output_dir, "downloads"), exist_ok=True)

    files_to_download = [
        ExtractInfo(
            file_path="/single_frames/" + entry.name,
            size=entry.size,
            url=url,
            output_dir=output_dir,
            rm=rm,
            dry_run=dry_run,
            extract=extract,
        )
        for entry in res.entries
        if _filter_entry(entry, filter_settings)
    ]

    if parallel:
        with ThreadPoolExecutor() as pool:
            pool.map(lambda info: _download_and_extract(dbx, info), files_to_download)
    else:
        for info in files_to_download:
            _download_and_extract(dbx, info)


if __name__ == "__main__":
    typer.run(download_zod)

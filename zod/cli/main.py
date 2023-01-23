from pathlib import Path
from typing import Tuple

try:
    import typer
except ImportError:
    print('zod is installed without the CLI dependencies: pip install "zod[cli]"')
    exit(1)

from zod.cli.download_zod import app as download_app
from zod.cli.generate_coco_json import convert_to_coco
from zod.cli.visualize_lidar import app as visualize_lidar_app

app = typer.Typer(help="Zenseact Open Dataset CLI.", no_args_is_help=True)

app.add_typer(download_app, name="download")

visualize_app = typer.Typer(help="Visualization tools", no_args_is_help=True)
app.add_typer(visualize_app, name="visualize")
visualize_app.add_typer(visualize_lidar_app, name="lidar")


convert_app = typer.Typer(
    help="Convert the Zenseact Open Dataset to a different format.", no_args_is_help=True
)


def tsr_dummy(
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
        file_okay=False,
        dir_okay=True,
        writable=True,
        readable=True,
        resolve_path=True,
        help="Path to the output directory.",
    ),
    version: str = typer.Option("full", help="Version of the dataset to use. One of: full, small."),
    path_size: Tuple[int, int] = typer.Option((64, 64), help="Path resultion."),
):
    typer.echo("Not Implemented Yet")


convert_app.command("coco", no_args_is_help=True)(convert_to_coco)
convert_app.command("tsr-patches", no_args_is_help=True)(tsr_dummy)
app.add_typer(convert_app, name="generate")


if __name__ == "__main__":
    app()

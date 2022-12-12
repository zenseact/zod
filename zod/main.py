from pathlib import Path
from typing import Tuple

import typer

from zod.frames.generate_coco_json import convert_to_coco
from zod.utils.download_zod import download_zod


app = typer.Typer(help="Zenseact Open Dataset CLI.", no_args_is_help=True)

app.command("download", no_args_is_help=True)(download_zod)


convert_app = typer.Typer(
    help="Convert the Zenseact Open Dataset to a different format.", no_args_is_help=True
)


def tsr_dummy(
    dataset_root: Path = typer.Option(..., help="Path to the root of the ZOD dataset."),
    output_dir: Path = typer.Option(..., help="Path to the output directory."),
    version: str = typer.Option("full", help="Version of the dataset to use. One of: full, small."),
    path_size: Tuple[int, int] = typer.Option((64, 64), help="Path resultion."),
):
    typer.echo("Not Implemented Yet")


convert_app.command("coco", no_args_is_help=True)(convert_to_coco)
convert_app.command("tsr-patches", no_args_is_help=True)(tsr_dummy)
app.add_typer(convert_app, name="generate")


if __name__ == "__main__":
    app()

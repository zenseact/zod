try:
    import typer
except ImportError:
    print('Warning! zod is installed without the CLI dependencies:\npip install "zod[cli]"')
    exit(1)

from zod.cli.download import app as download_app
from zod.cli.extract_tsr_patches import extract_tsr_patches
from zod.cli.generate_coco_json import convert_to_coco
from zod.cli.verify import app as verify_app
from zod.cli.visualize_lidar import app as visualize_lidar_app

visualize_app = typer.Typer(help="Visualize ZOD.", no_args_is_help=True)
visualize_app.add_typer(visualize_lidar_app, name="lidar")

convert_app = typer.Typer(help="Convert ZOD to a different format.", no_args_is_help=True)
convert_app.command("coco", no_args_is_help=True)(convert_to_coco)
convert_app.command("tsr-patches", no_args_is_help=True)(extract_tsr_patches)


def add_child(parent: typer.Typer, child: typer.Typer, name: str):
    """Workaround to handle single-command sub-apps."""
    if len(child.registered_commands) == 1:
        parent.command(name)(child.registered_commands[0].callback)
    else:
        parent.add_typer(child, name=name)


app = typer.Typer(help="Zenseact Open Dataset CLI.", no_args_is_help=True)

add_child(app, download_app, "download")
add_child(app, verify_app, "verify")
add_child(app, visualize_app, "visualize")
add_child(app, convert_app, "generate")


if __name__ == "__main__":
    app()

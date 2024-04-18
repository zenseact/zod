from typing import List

import numpy as np
import typer
from rich import print
from rich.panel import Panel

from zod import ZodFrames, ZodSequences
from zod.constants import AnnotationProject
from zod.data_classes import LidarData
from zod.data_classes.box import Box3D
from zod.utils.utils import zfill_id

app = typer.Typer(no_args_is_help=True)

IMPORT_ERROR = (
    "zod does not ship with open3d, which is required for interactive point cloud "
    "visualization.\nPlease install it manually: pip install open3d"
)


def _visualize(data: LidarData, boxes: List[Box3D] = None):
    try:
        import open3d as o3d
    except ImportError:
        print(Panel(IMPORT_ERROR, title="Warning! Missing Dependency", border_style="red"))
        exit(1)
    from matplotlib import cm

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data.points)
    # color the pointcloud according to the timestamp
    # timestamps = data.timestamps
    # color = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min())
    color = np.log(data.intensity)
    # use a nice colormap
    pcd.colors = o3d.utility.Vector3dVector(cm.get_cmap("jet")(color)[:, :3])

    # if boxes are provided, render them as rigid bodies
    if boxes:
        o3d_boxes = []
        for box in boxes:
            o3d_box = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(box.corners))
            o3d_box.color = (0.98, 0.63, 0.01)
            o3d_boxes.append(o3d_box)
        o3d.visualization.draw_geometries(
            [pcd, *o3d_boxes, o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)]
        )
    else:
        o3d.visualization.draw_geometries([pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)])


@app.command(no_args_is_help=True)
def frames(
    dataset_root: str = typer.Option(..., help="Path to the dataset root"),
    version: str = typer.Option(..., help="Version of the dataset"),
    frame_id: int = typer.Option(..., help="Frame id to visualize"),
    num_before: int = typer.Option(0, help="Number of frames before the given frame id"),
    num_after: int = typer.Option(0, help="Number of frames after the given frame id"),
    with_bounding_boxes: bool = typer.Option(False, help="if bounding boxes of center-frame are to be rendered"),
):
    """Visualize the lidar data for a given frame id."""
    zod_frames = ZodFrames(dataset_root=dataset_root, version=version)
    if zfill_id(frame_id) not in zod_frames.get_all_ids():
        raise ValueError(f"Frame id must be one of {zod_frames.get_all_ids()}.")
    frame = zod_frames[frame_id]
    data = frame.get_aggregated_lidar(num_before=num_before, num_after=num_after)
    if with_bounding_boxes:
        annos = frame.get_annotation(project=AnnotationProject.OBJECT_DETECTION)
        boxes = [anno.box3d for anno in annos if anno.box3d is not None]
    else:
        boxes = []
    _visualize(data, boxes=boxes)


@app.command(no_args_is_help=True)
def sequences(
    dataset_root: str = typer.Option(..., help="Path to the dataset root"),
    version: str = typer.Option(..., help="Version of the dataset"),
    sequence_id: int = typer.Option(..., help="Frame id to visualize"),
    start: int = typer.Option(0, help="Index of the first frame to visualize"),
    end: int = typer.Option(None, help="Index of the last frame to visualize (None means last)"),
    downsampling: int = typer.Option(10, help="Downsampling factor (random point dropping)"),
):
    """Visualize the lidar data for a given frame id."""
    zod_sequences = ZodSequences(dataset_root=dataset_root, version=version)
    if zfill_id(sequence_id) not in zod_sequences.get_all_ids():
        raise ValueError(f"Frame id must be one of {zod_sequences.get_all_ids()}.")
    frame = zod_sequences[sequence_id]
    data = frame.get_aggregated_lidar(start=start, end=end)
    if downsampling > 1:
        typer.echo(f"Will subsample the point-cloud with a factor {downsampling}")
        indexes = np.random.choice(data.points.shape[0], size=data.points.shape[0] // downsampling, replace=False)
        data.points = data.points[indexes]
        data.intensity = data.intensity[indexes]
        data.timestamps = data.timestamps[indexes]
        data.diode_idx = data.diode_idx[indexes]
    _visualize(data)


@app.command(no_args_is_help=True)
def path(
    path: str = typer.Argument(..., help="Path to the lidar data"),
):
    """Visualize a given lidar file (npy)."""
    _visualize(LidarData.from_npy(path))


if __name__ == "__main__":
    app()

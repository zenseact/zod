import numpy as np
import open3d as o3d
import typer
from matplotlib import cm

from zod import ZodFrames
from zod.utils.utils import zfill_id

app = typer.Typer(no_args_is_help=True)


def _visualize(points: np.ndarray, timestamps: np.ndarray, intensities: np.ndarray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # color the pointcloud according to the timestamp
    # color = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min())
    color = np.log(intensities)
    # use a nice colormap
    pcd.colors = o3d.utility.Vector3dVector(cm.get_cmap("jet")(color)[:, :3])
    # Draw coordinate frame
    o3d.visualization.draw_geometries(
        [pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)]
    )


@app.command(no_args_is_help=True)
def frames(
    dataset_root: str = typer.Option(..., help="Path to the dataset root"),
    version: str = typer.Option(..., help="Version of the dataset"),
    frame_id: int = typer.Option(..., help="Frame id to visualize"),
    num_before: int = typer.Option(0, help="Number of frames before the given frame id"),
    num_after: int = typer.Option(0, help="Number of frames after the given frame id"),
):
    """Visualize the lidar data for a given frame id."""
    zod_frames = ZodFrames(dataset_root=dataset_root, version=version)
    if zfill_id(frame_id) not in zod_frames.get_all_ids():
        raise ValueError(f"Frame id must be one of {zod_frames.get_all_ids()}.")
    frame = zod_frames[frame_id]
    data = frame.get_aggregated_point_cloud(num_before=num_before, num_after=num_after)
    _visualize(data.points, data.timestamps, data.intensity)


@app.command(no_args_is_help=True)
def path(
    path: str = typer.Argument(..., help="Path to the lidar data"),
):
    """Visualize the lidar data."""
    pc = np.load(path)
    # mask = (pc["diode_index"] > 128 + 16) & (pc["diode_index"] < 128 + 32)
    mask = pc["diode_index"] < 1280
    points = np.stack([pc["x"], pc["y"], pc["z"]], axis=1)[mask]
    timestamps = pc["timestamp"][mask]
    intensities = pc["intensity"][mask]
    _visualize(points, timestamps, intensities)


if __name__ == "__main__":
    app()

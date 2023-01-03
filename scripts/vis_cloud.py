import numpy as np
import open3d as o3d
import typer
from matplotlib import cm


def main(
    path: str = typer.Argument(..., help="Path to the lidar data"),
):
    """Visualize the lidar data."""
    pc = np.load(path)
    # mask = (pc["diode_index"] > 128 + 16) & (pc["diode_index"] < 128 + 32)
    mask = pc["diode_index"] < 1280
    points = np.stack([pc["x"], pc["y"], pc["z"]], axis=1)[mask]
    timestamps = pc["timestamp"][mask]
    intensities = pc["intensity"][mask]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # color the pointcloud according to the timestamp
    # color = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min())
    color = (intensities - intensities.min()) / (intensities.max() - intensities.min())
    # use a nice colormap
    pcd.colors = o3d.utility.Vector3dVector(cm.get_cmap("jet")(color)[:, :3])
    # Draw coordinate frame
    o3d.visualization.draw_geometries(
        [pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)]
    )


if __name__ == "__main__":
    typer.run(main)

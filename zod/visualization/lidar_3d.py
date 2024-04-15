"""3D visualization tool."""

from typing import Any, List, Tuple

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html

from zod import ZodFrames
from zod.constants import AnnotationProject
from zod.visualization.bev_utils import BEVSettings


def get_header(name):
    """Create an HTML header."""
    title = html.H4(name, style={"margin-top": 5})
    logo = html.Img(
        src="https://zenseact.com/wp-content/uploads/2022/10/Zenseact-logo-neg.svg",
        style={"float": "right", "height": 40, "margin-top": 5},
    )
    link = html.A(logo, href="https://zenseact.com/")

    return dbc.Row([dbc.Col(title, md=8), dbc.Col(link, md=4)], style={"height": 40})


class Viz3D:
    """3D Visualization class."""

    def __init__(self):
        """Initialize visualization figure."""
        self._settings = BEVSettings()
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
        self.app.title = "Zenseact Open Dataset Visualization"

        layout = go.Layout(
            template="plotly_dark",
            scene=dict(aspectmode="data"),
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(dtick=5, range=[-200, 200]),
            yaxis=dict(dtick=5, range=[-100, 100]),
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )

        self.fig = go.Figure(layout=layout)

    def add_pointcloud(self, x, y, z, intensity):
        """Add pointcloud to 3D graph."""
        self.fig.add_traces(
            data=[
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    name="LiDAR",
                    legendgroup="LiDAR",
                    mode="markers",
                    marker=dict(
                        size=1,
                        color=intensity,  # set color to an array/list of desired values
                        colorscale="viridis",  # choose a colorscale
                        opacity=1.0,
                    ),
                    showlegend=True,
                )
            ]
        )

    def add_zod_annotations(self, objects: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
        """Add 3D bounding boxed to figure."""
        class_names, positions, dimensions, rotations = objects
        n_objects = np.shape(positions)[0]

        if n_objects > 0:
            # Map classes
            classes = self._settings.encode_classes(class_names)
            for obj_idx in range(n_objects):
                self._add_box(
                    (
                        class_names[obj_idx],
                        classes[obj_idx],
                        positions[obj_idx],
                        dimensions[obj_idx],
                        rotations[obj_idx],
                    )
                )

    def _add_box(self, object_annotations: Tuple[Any], draw_edges: bool = False):
        """Construct 3D box mesh."""
        # pylint: disable-msg=too-many-locals

        # Extract object information
        class_name, class_ind, position, dimension, rotation = object_annotations

        coord1 = (-1) * dimension / 2
        coord3 = dimension / 2
        coord2 = np.vstack([coord1[0], coord3[1]])
        coord4 = np.vstack([coord3[0], coord1[1]])

        all_coords = np.vstack(
            [
                (rotation.rotate(coord1) + position),
                rotation.rotate(np.vstack([coord2, coord1[2]])) + position,
                rotation.rotate(coord3) + position,
                rotation.rotate(np.vstack([coord4, coord1[2]])) + position,
            ]
        ).T
        a = (coord1[2] + position[2]) * np.ones(4)
        b = (coord3[2] + position[2]) * np.ones(4)

        all_coords = np.hstack([np.vstack([all_coords[:2], a]), np.vstack([all_coords[:2], b])])

        colors_available = (
            px.colors.qualitative.Light24[:2]
            + px.colors.qualitative.Light24[4:8]
            + px.colors.qualitative.Light24[16:20]
            + px.colors.qualitative.Light24[-2:]
        )
        color = colors_available[int(class_ind)]

        # Boxes edges
        i = [7, 0, 0, 0, 4, 4, 6, 1, 4, 0, 3, 6]
        j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]
        k = [0, 7, 2, 3, 6, 7, 1, 6, 5, 5, 7, 2]

        if not draw_edges:
            self.fig.add_traces(
                [
                    go.Mesh3d(
                        # 8 vertices of a cube
                        x=all_coords[0, :],
                        y=all_coords[1, :],
                        z=all_coords[2, :],
                        i=i,
                        j=j,
                        k=k,
                        name=class_name,
                        legendgroup=class_name,
                        legendgrouptitle=dict(text=class_name),
                        showlegend=True,
                        opacity=0.2,
                        color=color,
                        flatshading=True,
                        contour=dict(color="white", show=True, width=10),
                        alphahull=0,
                    )
                ]
            )
        else:
            triangles = np.vstack((i, j, k)).T
            x = all_coords[0, :]
            y = all_coords[1, :]
            z = all_coords[2, :]
            vertices = np.vstack((x, y, z)).T
            tri_points = vertices[triangles]
            # extract the lists of x, y, z of the triangle vertices and connect them by a line
            xe = []
            ye = []
            ze = []
            for T in tri_points:
                xe.extend([T[k % 3][0] for k in range(4)] + [None])
                ye.extend([T[k % 3][1] for k in range(4)] + [None])
                ze.extend([T[k % 3][2] for k in range(4)] + [None])

            self.fig.add_traces(
                [
                    go.Scatter3d(
                        x=xe,
                        y=ye,
                        z=ze,
                        mode="lines",
                        name="",
                        line=dict(color=color, width=14),
                    ),
                    go.Scatter3d(
                        x=xe,
                        y=ye,
                        z=ze,
                        mode="lines",
                        name="",
                        line=dict(color="black", width=2),
                    ),
                ]
            )

    def show(self):
        """Render graph."""
        self.fig.update_layout(legend_title_text="Available Objects")
        self.app.layout = dbc.Container(
            [
                get_header(self.app.title),
                html.Hr(),
                dbc.Row(dcc.Graph(id="my-graph", figure=self.fig), style={"flexGrow": "1"}),
            ],
            fluid=True,
            style={"height": "100vh", "display": "flex", "flexDirection": "column"},
        )

        self.app.run_server(debug=False)


if __name__ == "__main__":
    #
    # Zenseact Open Dataset
    #
    # NOTE! Set the dataset path
    DATASET_ROOT = ""
    VERSION = "full"  # "mini" or "full"

    # initialize ZodFrames
    zod_frames = ZodFrames(dataset_root=DATASET_ROOT, version=VERSION)

    zod_frame = zod_frames[0]

    # get the LiDAR point cloud
    pcd = zod_frame.get_lidar()[0]

    # get the object annotations
    annotations = zod_frame.get_annotation(AnnotationProject.OBJECT_DETECTION)

    viz = Viz3D()
    viz.add_pointcloud(pcd.points[:, 0], pcd.points[:, 1], pcd.points[:, 2], pcd.intensity)
    viz.add_zod_annotations(
        (
            np.array([obj.name for obj in annotations if obj.box3d]),
            np.concatenate([obj.box3d.center[None, :] for obj in annotations if obj.box3d], axis=0),
            np.concatenate([obj.box3d.size[None, :] for obj in annotations if obj.box3d], axis=0),
            np.array([obj.box3d.orientation for obj in annotations if obj.box3d]),
        )
    )

    viz.show()

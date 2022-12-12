"""Utilities for creating point cloud input representation."""
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

# from constants import DO_CLASSES, DYNAMIC_OBJECTS, SO_CLASSES, STATIC_OBJECTS
# from plot_objects_annot_on_image import ObjectAnnotationHandler, get_annotations_files, read_anno_content


# AVAILABLE_PROJECTS = [DYNAMIC_OBJECTS, STATIC_OBJECTS]
# Dynamic Object classes for BEV visualization
DO_CLASSES = ("Vehicle", "VulnerableVehicle", "Pedestrian", "Animal")
# Static objects classes for BEV visualization
SO_CLASSES = (
    "TrafficSign",
    "PoleObject",
    "TrafficGuide",
    "TrafficBeacon",
    "TrafficSignal",
    "DynamicBarrier",
    "Inconclusive",
)


@dataclass
class BEVSettings:
    """All relevant information about the input."""

    # pylint: disable=too-many-instance-attributes
    # General settings
    grid_min: np.ndarray = np.array([-50.0, 0.0])
    grid_max: np.ndarray = np.array([50.0, 100.0])
    grid_cell_size: float = 0.1  # Default in PIXOR: 0.1

    # Pixor settings
    pixor_z_min: float = -2.4
    pixor_z_max: float = 1.0

    # === Not configurable (automatically calculated) ===
    # General
    grid_channels: int = 0
    grid_res: Tuple[int, int] = (0, 0)
    grid_shape: Tuple[int, int, int] = (0, 0, 0)

    classes: Tuple[str] = DO_CLASSES + SO_CLASSES

    def get_class_name(self, idx: int) -> str:
        """Retrieve class name from index."""
        return self.classes[idx]

    def encode_classes(self, classes_to_encode: List[str]) -> List[int]:
        """Retrieve classes indexes for a list of class names."""
        return [self.classes.index(entry) for entry in classes_to_encode]

    def __post_init__(self):
        """Post initialize fields."""
        # BEV Grid
        grid_res = (self.grid_max - self.grid_min) / self.grid_cell_size
        self.grid_res = tuple(grid_res.astype(int))
        self.grid_channels = int((self.pixor_z_max - self.pixor_z_min) / self.grid_cell_size) + 3
        self.grid_shape = (self.grid_channels,) + self.grid_res


def create_pointcloud_input(points: np.ndarray, settings: BEVSettings) -> np.ndarray:
    """Create input representation from raw data.

    Args:
        points: Point cloud [N, 4] containing ['x', 'y', 'z', 'intensity'].
        settings: Settings defining the input format.

    Returns:
        input_: An instance of the encoded input point cloud.

    """
    # Truncate points according to BEV ranges
    mask = get_grid_mask(points, settings)
    points = points[mask]

    point_indices_xy = get_grid_indices_xy(points[:, :3], settings)

    input_ = _create_pointcloud_input_pixor(points, point_indices_xy, settings)

    return input_


def _create_pointcloud_input_pixor(
    points: np.ndarray, point_indices_xy: np.ndarray, settings: BEVSettings
) -> np.ndarray:
    """Create PIXOR-style input representation.

    Args:
        points: Point cloud [N, 4] containing ['x', 'y', 'z', 'intensity'].
        point_indices_xy: The corresponding xy-indices for the points.
        settings: Settings defining the input format.

    Returns:
        A PIXOR style BEV projection of the input point cloud.

    """
    point_indices_c = np.cast["int32"](
        (points[:, 2] - settings.pixor_z_min) / settings.grid_cell_size
    )
    point_indices_c = 1 + np.clip(
        point_indices_c,
        a_min=-1,
        a_max=settings.grid_channels - 3,
    )
    point_indices_cxy = tuple(
        np.transpose(
            np.concatenate([np.expand_dims(point_indices_c, axis=-1), point_indices_xy], axis=-1)
        ).reshape(3, -1)
    )

    n_points = points.shape[0]

    point_indices_intensity_c = np.repeat(settings.grid_channels - 1, n_points)
    point_indices_intensity_cxy = tuple(
        np.transpose(
            np.concatenate(
                [np.expand_dims(point_indices_intensity_c, axis=-1), point_indices_xy], axis=-1
            )
        ).reshape(3, -1)
    )

    # Define the update per index (currently just occupancy)
    updates = np.ones((n_points,))
    updates_intensity = points[:, 3]

    # Create occupancy grid (with intensity)
    input_ = np.zeros(settings.grid_shape, dtype=np.float32)
    input_[point_indices_cxy] = updates  # Occupancy.
    input_[point_indices_intensity_cxy] = updates_intensity  # Intensity.

    return input_


def get_grid_mask(cloud: np.ndarray, settings: BEVSettings) -> np.ndarray:
    """Get the boolean mask to filter out points within the grid.

    Args:
        cloud: The input point cloud [N, 3]
        settings: The input definition.

    Returns:
        mask: Boolean mask containing true for points within grid.

    """
    scaled_xy = cloud[:, :2] - settings.grid_min
    bev_max = settings.grid_max - settings.grid_min
    mask = np.all(((scaled_xy >= 0) & (scaled_xy < bev_max)), axis=-1)
    return mask


def get_grid_indices_xy(cloud: np.ndarray, settings: BEVSettings) -> np.ndarray:
    """Get grid indices from point cloud.

    Args:
        cloud: Point cloud of x,y,z coordinates [N, 3].
        settings: Settings defining the grid size and resolution.

    Returns:
        indices_xy: The xy grid indices of each point [N, 2].

    """
    # Convert points to indices
    indices_xy = np.cast["int32"]((cloud[:, :2] - settings.grid_min) / settings.grid_cell_size)
    return indices_xy


def filter_point_cloud(cloud: np.ndarray, angle: np.ndarray, cam_pos: np.ndarray) -> np.ndarray:
    """Filter out points outside of camera-centered frustum."""
    cloud_xy = cloud[:, :2] - cam_pos[:2]
    point_angles = np.arctan2(cloud_xy[:, 1], cloud_xy[:, 0])
    mask = np.logical_or(point_angles < angle[0], point_angles > angle[1])
    return cloud[mask]


def get_objects_for_bev(
    seq_folder: str,
    annotation_projects: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get annotation objects for bird eye view visualization.

    Args:
        seq_folder: path to the sequence folder.
        annotation_projects: list of projects you want to visualize.
                             Possible projects: DYNAMIC_OBJECTS, STATIC_OBJECTS.

    Returns:
        extracted_anno_objects: positions, dimensions, rotations and classes of bounding boxes
                                to be visualized.

    """
    anno_project_files = get_annotations_files(seq_folder)
    anno_objects = []
    for proj in annotation_projects:
        if proj in AVAILABLE_PROJECTS:
            anno_file = anno_project_files[proj]
            anno_content = read_anno_content(anno_file)
            anno_objects.extend(list(ObjectAnnotationHandler.from_annotations(anno_content)))
        else:
            raise Exception(
                f"Project {proj} is not available to plot. "
                f"Available projects: {*AVAILABLE_PROJECTS,}."
            )
    positions = [obj[2].marking3d.get("Location")[:2] for obj in anno_objects if obj[2].marking3d]
    dimensions = [obj[2].marking3d.get("Size")[:2] for obj in anno_objects if obj[2].marking3d]
    rotations = [obj[2].marking3d.get("Rotation") for obj in anno_objects if obj[2].marking3d]
    classes = [[obj[0]] for obj in anno_objects if obj[2].marking3d]
    extracted_anno_objects = (
        np.array(classes),
        np.array(positions),
        np.array(dimensions),
        np.array(rotations),
    )
    return extracted_anno_objects

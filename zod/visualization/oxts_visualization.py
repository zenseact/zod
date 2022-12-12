"""Plot gps information."""

from typing import List, Optional, Sequence
import json
import numpy as np
import plotly.express as px

from ..constants import (
    DEFAULT_COLOR,
    DEFAULT_SIZE,
    MAPS_STYLE,
    OPACITY_LEVEL,
    SIZE_MAX,
)


class PlotlyAutoZoomer:
    """Class to find zoom level automatically based on long / lat diff as reference."""

    LONG_LAT_DIFFS = (0.00001, 0.0001, 0.0005, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 3)
    ZOOM_LEVELS = (24, 20, 18, 17, 14, 12, 11, 10, 9, 8, 7, 6, 4)

    @classmethod
    def calc_zoom_level(cls, lats: List[float], longs: List[float]) -> int:
        """Calculate zoom level based on lats and lons values.

        Args:
            lats : list of latitudes
            longs : list of longitudes

        Returns:
            zoom level for plotly

        """
        lats = np.array(lats)
        longs = np.array(longs)
        lat_diff = lats.max() - lats.min()
        long_diff = longs.max() - longs.min()
        max_diff = max(lat_diff, long_diff)
        return np.round(np.interp(max_diff, cls.LONG_LAT_DIFFS, cls.ZOOM_LEVELS))


def plot_gps(
    longs: Sequence[float],
    lats: Sequence[float],
    colours: Optional[List[str]] = None,
    sizes: Optional[List[int]] = None,
    show: bool = False,
):
    """Plot gps coordinates on map.

    Args:
        longs : gps points longitudes
        lats : gps points latitudes
        colours : colour of points on the map
        sizes : size of points on the map
        show : show map on browser if True, otherwise wait for further plots

    """
    assert len(longs) == len(lats)
    if not colours:
        colours = [DEFAULT_COLOR] * len(longs)
    if not sizes:
        sizes = [DEFAULT_SIZE] * len(longs)
    fig = px.scatter_mapbox(
        lat=lats,
        lon=longs,
        color=colours,
        size=sizes,
        zoom=PlotlyAutoZoomer.calc_zoom_level(lats, longs),
        size_max=SIZE_MAX,
        opacity=OPACITY_LEVEL,
    )
    fig.update_layout(mapbox_style=MAPS_STYLE)
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    if show:
        fig.show()


def plot_gps_track_from_dataset_sequence(oxts_values: np.ndarray):
    """Plot GPS track on the map from dataset sequence.

    Args:
        oxts_values : OxTS values

    """
    longs, lats = oxts_values.pos_lon, oxts_values.pos_lat
    colors = ["blue"] * len(longs)
    sizes = [1] * len(longs)
    plot_gps(longs, lats, colors, sizes, show=True)


def show_gps_for_all_frames(filename: str):
    """Show GPS points for all extracted frames in dataset.

    Args:
        filename: path to JSON file containing GPS coordinates for all frames in dataset

    """
    with open(filename) as opened:
        content = json.load(opened)
    lats, lons = [], []
    vehicles = set()
    for frame_id, points in content.items():
        vehicles.add(frame_id.split("_")[0])
        lons.append(points[0])
        lats.append(points[1])
    print(
        f"Total frames in dataset: {len(content)}, vehicles: {vehicles}",
    )
    plot_gps(lons, lats, ["blue"] * len(lons), [1] * len(lons), show=True)

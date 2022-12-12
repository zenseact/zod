"""Module to perform OxTS extraction and visualize GPS track projection on image plane."""
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import pyproj
from pytz import utc

DEFAULT_COL_VALUES = {
    "pitchMissalignment": (0, 0, 0, b"Radians"),
    "headingMissalignment": (0, 0, 0, b"Radians"),
}
ECEF_XYZ = ["ecef_x", "ecef_y", "ecef_z"]
OXTS_OPTIONAL_COLS = [*DEFAULT_COL_VALUES.keys()]
OXTS_COLS = [
    "undulation",
    "timestamp",
    "posLat",
    "posLon",
    "posAlt",
    "heading",
    "pitch",
    "roll",
    "velForward",
    "velDown",
    "velLateral",
    "leapSeconds",
]

# pylint: disable=C0103
ECEF = pyproj.Proj(proj="geocent", ellps="WGS84", datum="WGS84")
LLA = pyproj.Proj(proj="latlong", ellps="WGS84", datum="WGS84")
GPS_EPOCH = datetime(1980, 1, 6, tzinfo=utc)
PATH_POINTS = np.arange(5, 201, 5)


def move(T, P, *, vec_dim=None):
    """Apply transform to points.

    Notes
    -----
    If the dimensionality of `P` is known, and you have a very large batch
    of `T` or `P`, consider passing the `vec_dim` argument to the function.
    This will switch to a codepath that depending on the situation can be
    up to 1000x faster to compute.

    """
    R, t = R_t(T)
    return matvec(R, P, nrow=vec_dim) + t


def T_from_R_t(R, t):
    """Combine rotation matrix and translation vec into a transform."""
    t = t[..., None]
    o = np.zeros_like(R[..., -1:, :])
    i = np.ones_like(t[..., -1:, :])
    return np.concatenate(
        [np.concatenate([R, t], axis=-1), np.concatenate([o, i], axis=-1)], axis=-2
    )


def T_inv(T, *, vec_dim=None):
    """Compute inverse transform.

    Notes
    -----
    This computes an inverse of a transform matrix of form

    ```
                  T   T
    inv |R t| = |R  -R t|
        |0 1|   |0    1 |
    ```

    If the dimension of the problem (for the matrix `T` of shape NxN the
    the dimension is N-1) is known, and you have a very large batch of `T`,
    consider passing the `vec_dim` argument to the function. This will
    switch to a codepath that depending on the situation can be up to 1000x
    faster to compute.

    """
    R, t = R_t(T)
    return T_from_R_t(mT(R), -matvec(mT(R), t, nrow=vec_dim))


def mT(M):
    """Compute matrix transpose in the last two dimensions."""
    return np.einsum("...ij->...ji", M)


def Ry(r):
    """Construct a rotation matrix around Y-axis."""
    return _Ry(np.cos(r), np.sin(r))


def Rz(r):
    """Construct a rotation matrix around Z-axis."""
    return _Rz(np.cos(r), np.sin(r))


def Rx(r):
    """Construct a rotation matrix around X-axis."""
    return _Rx(np.cos(r), np.sin(r))


def R_t(T):
    """Decompose a transform into rotation matrix and translation vec."""
    return T[..., :-1, :-1], T[..., :-1, -1]


def _Rx(c, s):
    """Construct a rotation matrix around X-axis given cos and sin.

    The `c` and `s` MUST satisfy c^2 + s^2 = 1 and have the same shape.

    See https://en.wikipedia.org/wiki/Rotation_matrix#Basic_rotations.

    """
    o = np.zeros_like(c)
    i = np.ones_like(o)
    return _tailstack2([[i, o, o], [o, c, -s], [o, s, c]])


def _Ry(c, s):
    o = np.zeros_like(c)
    i = np.ones_like(o)
    return _tailstack2([[c, o, s], [o, i, o], [-s, o, c]])


def _Rz(c, s):
    o = np.zeros_like(c)
    i = np.ones_like(o)
    return _tailstack2([[c, -s, o], [s, c, o], [o, o, i]])


def rotation_matrix(roll, pitch, yaw):
    """Compute extrinsic x-y-z (intrinsic z-y'-x'') rotation matrix.

    This works e.g. for converting ENU/NED to ISO/OXTS vehicle frame
    given roll, pitch and yaw/heading angles.

    See https://support.oxts.com/hc/en-us/articles/
    115002859149-OxTS-Reference-Frames-and-ISO8855-Reference-Frames
    for more context.

    WARNING: If you are not sure this function works for your use case,
    consider composing rotations yourself using functions `Rx`, `Ry` and
    `Rz`.

    """
    return Rz(yaw) @ Ry(pitch) @ Rx(roll)


def _tailstack2(nxm):
    """Stack a list of lists of tensors in the tail dimensions.

    Take list of length N of lists of length M of arrays of shape (...),
    and return an array of shape (..., N, M). This does not broadcast,
    so the shape (...) MUST be the same for all arrays.

    """
    # pylint: disable=unnecessary-comprehension
    return np.stack([np.stack([m for m in n], axis=-1) for n in nxm], axis=-2)


def to_unit_norm(a):
    """Normalize (batch of) vectors to unit norm in the last dimension."""
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def to_homogenous(a):
    """Append ones to the last dimension of `a`."""
    return np.concatenate([a, np.ones_like(a[..., 0:1])], axis=-1)


def matvec(A, b, *, nrow=None):
    """Multiply vector by matrix.

    Notes
    -----
    If you have very large batches of small-dimensional
    matrices / vectors, you should consider passing the
    number of rows in the matrix with the `nrow` argument,
    as this will switch to a codepath that depending on
    the situation can be up to 1000x faster to compute.

    """
    if nrow is not None:
        return np.stack([inner(A[..., i, :], b) for i in range(nrow)], axis=-1)
    return (A @ b[..., None])[..., 0]


def inner(a, b):
    """Compute broadcastable inner product in the last dimension."""
    return np.sum(a * b, axis=-1)


def kannala_project(P, K, dist):
    """Project 3D -> pixel coordinates under the Kannala camera model.

    Parameters
    ----------
    P: tensor of shape `(..., 3)`
        3D coordinates
    K: tensor of shape `(..., 3, 3)`
        Camera matrix
    dist: tensor of shape `(..., 4)`
        Distortion coefficients

    Returns
    -------
    p: tensor of shape `(..., 2)`
        Projected pixel coordinates

    """
    xy = P[..., :2]
    radius = np.linalg.norm(xy, axis=-1, keepdims=True)
    theta = np.arctan2(radius, P[..., 2:3])
    dist_angle = theta * (
        1
        + dist[..., 0:1] * theta**2
        + dist[..., 1:2] * theta**4
        + dist[..., 2:3] * theta**6
        + dist[..., 3:4] * theta**8
    )

    uv_distorted = dist_angle * np.where(radius != 0, xy / radius, np.zeros_like(xy))
    uv_hom = to_homogenous(uv_distorted)

    return matvec(K[..., :2, :], uv_hom, nrow=2)


def convert_h5_to_pandas(oxts_h5):
    """Convert HDF5 OXTS output into a Pandas DataFrame."""

    def column_from_samples(oxts_samples, col):
        return [
            sample[col]
            if col not in OXTS_OPTIONAL_COLS
            else (sample[col] if col in sample.dtype.names else DEFAULT_COL_VALUES[col])
            for sample in oxts_samples
        ]

    oxts_dict = {col: column_from_samples(oxts_h5, col) for col in OXTS_COLS + OXTS_OPTIONAL_COLS}
    oxts_dataframe = pd.DataFrame.from_dict(oxts_dict).set_index("timestamp")
    return oxts_dataframe


def convert_lla_to_ecef(lat_deg, lon_deg, alt):
    """Convert LLA to ECEF and return separately X, Y, Z coords."""
    ecef_x, ecef_y, ecef_z = pyproj.transform(LLA, ECEF, lon_deg, lat_deg, alt, radians=False)
    return ecef_x, ecef_y, ecef_z


def preprocess_oxts(oxts):
    """Convert the raw oxts frame into convenient format.

    Make altitude ellipsoidal.
    Pre-compute ECEF coordinates for the oxts frame.
    Pre-compute traveled distance.
    Fix pitch.
    Compute UTC timestamps from GPS time.

    """
    # Undulation is the difference between the WGS84 elipsoid model
    # of the earth and the altitude on the earth surface. At the sea
    # level, undulation values are negligible.
    # This value is generated by the oxts reciever using the
    # altitude table EGM96
    alt_ellips = oxts.posAlt.values + oxts.undulation.values
    ecef_x, ecef_y, ecef_z = convert_lla_to_ecef(oxts.posLat.values, oxts.posLon.values, alt_ellips)
    ecef = np.stack([ecef_x, ecef_y, ecef_z], axis=-1)
    traveled = np.cumsum(np.linalg.norm(ecef[1:] - ecef[:-1], axis=-1))
    traveled = np.concatenate([[0], traveled])

    def misalignment_to_float(field):
        value, _, _, units = field
        if units == b"Radians":
            return np.degrees(value)
        return np.array(value)

    # pylint: disable=invalid-name
    def apply_misalignment_to_column(o, column):
        try:
            return o[column] + o[f"{column}Missalignment"].map(misalignment_to_float)
        except KeyError:  # This happens for newly preprocessed drives
            return o[column]

    return (
        oxts.reset_index()
        .rename(columns={"timestamp": "time_gps"})
        .assign(
            posAlt=alt_ellips,
            ecef_x=ecef_x,
            ecef_y=ecef_y,
            ecef_z=ecef_z,
            traveled=traveled,
            pitch=lambda o: apply_misalignment_to_column(o, "pitch"),
            heading=lambda o: apply_misalignment_to_column(o, "heading"),
            time_utc=lambda o: GPS_EPOCH + pd.to_timedelta(o.time_gps + o.leapSeconds, unit="s"),
        )
        .drop(columns=["undulation"])
        .drop(columns=["pitchMissalignment", "headingMissalignment"], errors="ignore")
        .set_index("time_utc")
    )


def get_initial_position(oxts, time_utc):
    """Interpolate between OXTS points nearest to a UTC time."""
    prev_idx = oxts.index.get_loc(time_utc, method="ffill")
    prev_row = oxts.iloc[prev_idx]
    prev_time = oxts.index[prev_idx].timestamp()
    next_row = oxts.iloc[prev_idx + 1]
    next_time = oxts.index[prev_idx + 1].timestamp()
    this_time = time_utc.timestamp()
    alpha = (this_time - next_time) / (prev_time - next_time)
    return (alpha * prev_row + (1 - alpha) * next_row).rename(time_utc)


def _idx_of(iterable, condition=bool):
    return next((i for i, e in enumerate(iterable) if condition(e)), None)


def interpolate_minus_plus_oxts(oxts_minus, oxts_plus, start_traveled):
    """Interpolate between closer and further points to get path point estimates."""
    traveled_minus = oxts_minus.traveled.values - start_traveled
    traveled_plus = oxts_plus.traveled.values - start_traveled
    alpha = (PATH_POINTS - traveled_plus) / (traveled_minus - traveled_plus)
    new_idx = oxts_minus.index + (1 - alpha) * (oxts_plus.index - oxts_minus.index)
    return alpha[..., None] * oxts_minus.set_index(new_idx) + (
        1 - alpha[..., None]
    ) * oxts_plus.set_index(new_idx)


def _find_oxts_path_points(oxts, frame_time_utc):
    # Find start of the relevant oxts portion
    oxts_0 = get_initial_position(oxts, frame_time_utc)
    start_idx = oxts.index.get_loc(frame_time_utc, method="ffill")
    oxts_onward = oxts.iloc[start_idx:]

    # Find the end of the relevant oxts portion
    start_traveled = oxts_0.traveled
    # This will throw a ValueError if the travel distance of
    # PATH_POINTS[-1] from the oxts_0 is not reached until the
    # end of the `oxts` frame
    end_idx = _idx_of(oxts_onward.traveled.values, lambda x: x > start_traveled + PATH_POINTS[-1])
    if end_idx is None:
        raise ValueError(
            f"The OXTS segment ends less than {PATH_POINTS[-1]} m "
            f"away from the starting position at {frame_time_utc}."
        )
    oxts_segment = oxts_onward.iloc[: end_idx + 1]

    # Find path point candidates in the selected oxts portion
    path_points_idxs = np.argmax(
        oxts_segment.traveled.values[..., None] > start_traveled + PATH_POINTS, axis=0
    )

    if len(np.unique(path_points_idxs)) < len(PATH_POINTS):
        raise ValueError(
            "There are some path points missing in the OXTS segment. "
            "There is possibly a jump in GPS coordinates."
        )

    # Combine with previous points of candidates and interpolate
    oxts_path_points = interpolate_minus_plus_oxts(
        oxts_segment.iloc[path_points_idxs - 1], oxts_segment.iloc[path_points_idxs], start_traveled
    )

    return oxts_0, oxts_path_points


def ecef_to_enu_rotation(lat_deg, lon_deg):
    """Compute rotation matrix from ECEF to ENU at given coordinates."""
    sl = np.sin(np.radians(lat_deg))
    cl = np.cos(np.radians(lat_deg))
    sp = np.sin(np.radians(lon_deg))
    cp = np.cos(np.radians(lon_deg))

    o = np.zeros_like(sl)

    return np.stack(
        [
            np.stack([-sp, cp, o], axis=-1),
            np.stack([-cp * sl, -sp * sl, cl], axis=-1),
            np.stack([cp * cl, sp * cl, sl], axis=-1),
        ],
        axis=-2,
    )


def nwu_to_ref_rotation_from_points(points_x_nwu):
    """Compute matrix with heading and pitch by aligning path points."""

    def c_s_from_a_b(a, b):
        c = inner(a, b)
        s = np.cross(a, b)
        return c, s

    heading_nwu = to_unit_norm(points_x_nwu[..., [0, 1]])
    R_head = _Rz(*c_s_from_a_b(heading_nwu, np.array([1, 0])))

    points_x_fl = matvec(R_head, points_x_nwu)
    pitch_fl = to_unit_norm(points_x_fl[..., [0, 2]])

    _cp, _sp = c_s_from_a_b(pitch_fl, np.array([1, 0]))
    R_pitch = _Ry(_cp, -_sp)

    return R_pitch @ R_head


def _find_pitch_at_point(point, oxts):
    def median_direction(a):
        return to_unit_norm(np.median(to_unit_norm(a).reshape(-1, a.shape[-1]), axis=0))

    enu_R_ecef = ecef_to_enu_rotation(point.posLat, point.posLon)
    nwu_R_enu = rotation_matrix(0, 0, np.radians(-90))
    nwu_R_ecef = nwu_R_enu @ enu_R_ecef

    relevant_oxts = oxts[(oxts.traveled > point.traveled) & (oxts.traveled < point.traveled + 2)]
    nwu = matvec(nwu_R_ecef, relevant_oxts[ECEF_XYZ].values - point[ECEF_XYZ].values)
    ref_R_nwu = nwu_to_ref_rotation_from_points(median_direction(nwu))
    sy = ref_R_nwu[..., 0, 2]
    cy = ref_R_nwu[..., 2, 2]
    return np.degrees(np.arctan2(sy, cy))


def enu_to_ref_frame_rotation(heading_deg, pitch_deg, roll_deg):
    """Compute rotation matrix from ISO 8855 ENU to vehicle reference frame.

    Refer to https://support.oxts.com/hc/en-us/articles/
    115002859149-OxTS-Reference-Frames-and-ISO8855-Reference-Frames

    """
    o = np.zeros_like(heading_deg)
    i = np.ones_like(o)
    A = rotation_matrix(o, o, np.radians(90 * i))
    B = np.eye(3)
    C = rotation_matrix(np.radians(180 * i), o, o)
    HPR = rotation_matrix(np.radians(roll_deg), np.radians(pitch_deg), np.radians(heading_deg))

    return mT(C) @ mT(HPR) @ A @ B @ C


# pylint: disable=too-many-arguments
def ecef_to_ref_frame_transform(lat_deg, lon_deg, heading_deg, pitch_deg, roll_deg, ecef_xyz):
    """Compute 4x4 transformation from ECEF to OXTS reference frame."""
    enu_R_ecef = ecef_to_enu_rotation(lat_deg, lon_deg)
    ref_R_enu = enu_to_ref_frame_rotation(heading_deg, pitch_deg, roll_deg)
    ref_R_ecef = ref_R_enu @ enu_R_ecef
    ecef_t_ref = ecef_xyz
    ref_t_ecef = -matvec(ref_R_ecef, ecef_t_ref)
    return T_from_R_t(ref_R_ecef, ref_t_ecef)


# pylint: disable=invalid-name
def odometry_from_oxts(oxts, oxts_0=None):
    """Compute 4x4 odometry transform matrices relative to reference position.

    If `oxts_0` is a pandas.Series like a row in `oxts` dataframe, then it
    will be used as the reference position. If it's `None`, the first row
    `oxts.iloc[0]` will be used for that.

    """
    if oxts_0 is None:
        oxts_0 = oxts.iloc[0]

    ref_T_ecef = ecef_to_ref_frame_transform(
        oxts.posLat.values,
        oxts.posLon.values,
        oxts.heading.values,
        oxts.pitch.values,
        oxts.roll.values,
        oxts[ECEF_XYZ].values,
    )
    ref0_T_ecef = ecef_to_ref_frame_transform(
        oxts_0.posLat,
        oxts_0.posLon,
        oxts_0.heading,
        oxts_0.pitch,
        oxts_0.roll,
        oxts_0[ECEF_XYZ].values,
    )
    ref0_T_ref = ref0_T_ecef @ T_inv(ref_T_ecef)
    return ref0_T_ref


def generate_odometry(oxts, frame_time_utc, pitch_from_points=False):
    """Given oxts log generate car odometry relative to frame.

    Args:
        oxts(pd.DataFrame):
            OXTS dataframe (e.g. coming from
            query_oxts -> convert_h5_to_pandas -> preprocess_oxts)
        frame_time_utc(datetime):
            Time for the initial path point. Must have UTC timezone.
        pitch_from_points(bool):
            Indicates whether to compute pitch from neighboring points,
            or to take the values from the oxts frame.

    Returns:
        array of shape (len(PATH_POINTS), 4, 4) with odometry relative to
        the frame at PATH_POINTS travel distances from the frame.

    """
    oxts_0, oxts_path_points = _find_oxts_path_points(oxts, frame_time_utc)

    if pitch_from_points:
        oxts_0 = oxts_0.copy()  # in case someone needs the original
        oxts_0.pitch = _find_pitch_at_point(oxts_0, oxts)
        oxts_path_points = oxts_path_points.assign(
            pitch=[_find_pitch_at_point(point, oxts) for _, point in oxts_path_points.iterrows()]
        )

    # Compute odometry relative to oxts_0
    odometry = odometry_from_oxts(oxts_path_points, oxts_0)
    return odometry


def get_path_from_oxts(oxts_h5: np.ndarray, frame_time_utc: datetime):
    """Demonstrate how to get a HP GT path from only a frame_id.

    Args:
        oxts_h5: OxTS data for sequence
        frame_time_utc:

    """
    oxts_dataframe = convert_h5_to_pandas(oxts_h5)
    preprocessed_oxts = preprocess_oxts(oxts_dataframe)
    odometry = generate_odometry(preprocessed_oxts, frame_time_utc)
    return R_t(odometry)[1]


def draw_line(image, line, color):
    """Draw a line in image."""
    return cv2.polylines(
        image.copy(), [np.round(line).astype(np.int32)], isClosed=False, color=color, thickness=10
    )


def _get_path_in_cam(path: np.ndarray, calib: dict):
    return kannala_project(
        move(T_inv(calib.extrinsics), path - [0, 0, 0.3]),
        calib.intrinsics,
        calib.distortion,
    )


def visualize_gps_on_image(
    oxts_data: np.ndarray, frame_time: datetime, calib: dict, image: np.ndarray
):
    """Visualize GPS track on image."""
    path_3d = get_path_from_oxts(oxts_data, frame_time)
    path_on_image = _get_path_in_cam(path_3d, calib)
    image = draw_line(image, path_on_image, (50, 100, 200))
    return image

"""ZOD Frames module."""
import json
import os
from datetime import datetime
from itertools import repeat
from pathlib import Path
from typing import Dict, List, Tuple, Union

import h5py
import numpy as np
from pytz import utc
from tqdm.contrib.concurrent import process_map

from zod import constants
from zod.frames.annotation_parser import parse_ego_road_annotation, parse_lane_markings_annotation
from zod.utils.geometry import _interpolate_oxts_data
from zod.utils.objects import AnnotatedObject
from zod.utils.utils import gps_time_to_datetime, parse_timestamp_from_filename
from zod.utils.zod_dataclasses import (
    Calibration,
    FrameInformation,
    LidarData,
    MetaData,
    OXTSData,
    Pose,
)
from zod.visualization.oxts_on_image import _odometry_from_oxts


def _create_frame(frame: dict, dataset_root: str) -> FrameInformation:
    frame_info = FrameInformation.from_dict(frame)
    frame_info.convert_paths_to_absolute(dataset_root)
    return frame_info


def _get_frame_id(frame_id: Union[int, str]) -> str:
    if isinstance(frame_id, int):
        frame_id = str(frame_id).zfill(6)
    elif isinstance(frame_id, str):
        frame_id = frame_id.zfill(6)
    return frame_id


def _get_lidar_frames_path(
    core_frame_path: str, n_sweeps_before: int, n_sweeps_after: int
) -> List[str]:
    if n_sweeps_before == 0 and n_sweeps_after == 0:
        return [core_frame_path]

    surrounding_frames = sorted(os.listdir(os.path.dirname(core_frame_path)))
    core_frame_idx = surrounding_frames.index(os.path.basename(core_frame_path))
    frame_paths = [
        os.path.join(os.path.dirname(core_frame_path), f)
        for f in surrounding_frames[
            max(0, core_frame_idx - n_sweeps_before) : core_frame_idx + n_sweeps_after + 1
        ]
    ]
    return frame_paths


class ZodFrames(object):
    """ZOD Frames."""

    def __init__(self, dataset_root: Union[Path, str], version: str):
        self._dataset_root = dataset_root
        self._version = version
        assert (
            version in constants.VERSIONS
        ), f"Unknown version: {version}, must be one of: {constants.VERSIONS}"
        self._train_frames, self._val_frames = self._load_frames()
        self._frames: Dict[str, FrameInformation] = {**self._train_frames, **self._val_frames}

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, frame_id: Union[int, str]) -> FrameInformation:
        """Get frame by id, which is zero-padded frame number."""
        frame_id = _get_frame_id(frame_id)
        return self._frames[frame_id]

    def _load_frames(self) -> Tuple[Dict[str, FrameInformation], Dict[str, FrameInformation]]:
        """Load frames for the given version."""
        filename = constants.SPLIT_TO_TRAIN_VAL_FILE_SINGLE_FRAMES[self._version]

        with open(os.path.join(self._dataset_root, filename), "r") as f:
            all_ids = json.load(f)

        train_frames = process_map(
            _create_frame,
            all_ids[constants.TRAIN],
            repeat(self._dataset_root),
            desc="Loading train frames",
            chunksize=50 if self._version == constants.FULL else 1,
        )
        val_frames = process_map(
            _create_frame,
            all_ids[constants.VAL],
            repeat(self._dataset_root),
            desc="Loading val frames",
            chunksize=50 if self._version == constants.FULL else 1,
        )

        train_frames = {frame.frame_id: frame for frame in train_frames}
        val_frames = {frame.frame_id: frame for frame in val_frames}

        return train_frames, val_frames

    def get_split(self, split: str) -> List[str]:
        """Get split by name (e.g. train / val)."""
        if split == constants.TRAIN:
            return list(self._train_frames.keys())
        elif split == constants.VAL:
            return list(self._val_frames.keys())
        else:
            raise ValueError(
                f"Unknown split: {split}, should be {constants.TRAIN} or {constants.VAL}"
            )

    def get_all_frames(self) -> Dict[str, FrameInformation]:
        """Get all frames."""
        return self._frames

    def get_camera_frame(
        self, frame_id: Union[int, str], anonymization_method: str = "blur", camera: str = "front"
    ) -> str:
        """Get camera frame with anonymization_method either "blur"
        or "dnat" from camera "front" or *not yet available*"""
        frame_id = _get_frame_id(frame_id)
        if anonymization_method not in ["blur", "dnat"]:
            raise ValueError("Not a valid anonymization method")
        return self._frames[frame_id].camera_frame["camera_" + camera + "_" + anonymization_method]

    def get_image_path(
        self, frame_id: Union[int, str], anonymization_method: str = "blur", camera: str = "front"
    ) -> str:
        """Get camera frame with anonymization_method either "blur"
        or "dnat" from camera "front" or *not yet available*"""
        if anonymization_method not in ["blur", "dnat"]:
            raise ValueError("Not a valid anonymization method")
        return (
            self._frames[_get_frame_id(frame_id)]
            .camera_frame["camera_" + camera + "_" + anonymization_method]
            .filepath
        )

    def get_lane_markings_annotation_path(self, frame_id: Union[int, str]) -> str:
        """Get the path to lane markings annotation"""
        return self._frames[_get_frame_id(frame_id)].lane_markings_annotation_path

    def get_ego_road_annotation_path(self, frame_id: Union[int, str]) -> str:
        """Get the path to ego road annotation"""
        return self._frames[_get_frame_id(frame_id)].ego_road_annotation_path

    def get_timestamp(self, frame_id: Union[int, str]):
        return self._frames[_get_frame_id(frame_id)].timestamp

    def read_lane_markings_annotation(self, frame_id: Union[int, str]):
        p = self.get_lane_markings_annotation_path(frame_id)
        return parse_lane_markings_annotation(p)

    def read_ego_road_annotation(self, frame_id: Union[int, str]):
        p = self.get_ego_road_annotation_path(frame_id)
        return parse_ego_road_annotation(p)

    def read_calibration(self, frame_id: str) -> Calibration:
        """Read calibration files from json format."""
        with open(self._frames[frame_id].calibration_path) as f:
            calib = json.load(f)
        return Calibration.from_dict(calib)

    def read_meta_data(self, frame_id: str) -> MetaData:
        """Read meta data files from json format."""
        with open(self._frames[frame_id].metadata_path) as f:
            meta_data = json.load(f)
        return MetaData.from_dict(meta_data)

    def read_oxts(self, frame_id: str) -> OXTSData:
        """Read OXTS files from hdf5 format."""
        with h5py.File(self._frames[frame_id].oxts_path, "r") as f:
            data = OXTSData.from_hdf5(f)
        return data

    def read_object_detection_annotation(self, frame_id: str) -> List[AnnotatedObject]:
        """Read object detection annotation from json format."""
        with open(self._frames[frame_id].object_detection_annotation_path) as f:
            objs = json.load(f)
        return [AnnotatedObject.from_dict(anno) for anno in objs]

    def aggregate_pointclouds(
        self, frame_id: str, lidar_name: str, n_sweeps_before: int = 0, n_sweeps_after: int = 0
    ) -> LidarData:
        """Aggregate pointclouds from npy format."""
        # Read relevant data
        oxts = self.read_oxts(frame_id)
        lidar_pose = self.read_calibration(frame_id).lidars[lidar_name].extrinsics

        # Get the surrounding frames
        frame_paths = _get_surrounding_lidar_frames_path(
            self._frames[frame_id].lidar_frame[lidar_name].filepath,
            n_sweeps_before,
            n_sweeps_after,
        )

        # get timestamp from filename
        timestamps = [parse_timestamp_from_filename(f) for f in frame_paths]

        # interpolate oxts data for each frame
        oxts_datetime = np.vectorize(
            lambda gps_time, leap_seconds: gps_time_to_datetime(
                gps_time * constants.MICROSEC_PER_SEC, leap_seconds
            )
        )(oxts.time_gps, oxts.leap_seconds)
        interp_oxts = _interpolate_oxts_data(oxts, oxts_datetime, timestamps)

        # get odometry
        odometry = _odometry_from_oxts(interp_oxts, interp_oxts.get_idx(n_sweeps_before))

        for i, frame_path in enumerate(frame_paths):
            lidar_data = LidarData.from_npy(frame_path)
            # project to ego vehicle frame using calib
            lidar_data.transform(lidar_pose)
            # project to world frame using odometry
            lidar_data.transform(Pose(odometry[i]))
            # project back to lidar frame using calib
            lidar_data.transform(lidar_pose.inverse)
            if i == 0:
                aggregated_lidar_data = lidar_data
            else:
                aggregated_lidar_data.append(lidar_data)

        return aggregated_lidar_data

    def motion_compensate_pointcloud(
        self, lidar_data: LidarData, frame_id: str, lidar_name: str
    ) -> LidarData:
        """Motion compensate pointclouds from npy format."""
        # Read relevant data
        oxts = self.read_oxts(frame_id)
        lidar_pose = self.read_calibration(frame_id).lidars[lidar_name].extrinsics

        # get timestamp for each point
        timestamps = np.vectorize(lambda x: datetime.fromtimestamp(x / constants.MICROSEC_PER_SEC))(
            lidar_data.timestamps
        )
        # get frame timestamp
        frame_timestamp = self.get_timestamp(frame_id).replace(tzinfo=None)
        offset = np.vectorize(
            lambda x: (x - frame_timestamp).total_seconds() * constants.MICROSEC_PER_SEC
        )(timestamps)
        closest_idx = abs(offset).argmin()

        # interpolate oxts data for each frame
        oxts_datetime = np.vectorize(
            lambda gps_time, leap_seconds: gps_time_to_datetime(
                gps_time * constants.MICROSEC_PER_SEC, leap_seconds
            )
        )(oxts.time_gps, oxts.leap_seconds)
        interp_oxts = _interpolate_oxts_data(oxts, oxts_datetime, timestamps)

        # get odometry
        odometry = _odometry_from_oxts(interp_oxts, interp_oxts.get_idx(closest_idx))

        # project to ego vehicle frame using calib
        lidar_data.transform(lidar_pose)
        # project to center frame using odometry
        rotations = odometry[:, :3, :3]
        translations = odometry[:, :3, 3]
        lidar_data.points = (
            lidar_data.points[:, None, :] @ rotations.transpose(0, 2, 1) + translations[:, None, :]
        ).squeeze(1)
        # lidar_data.transform(Pose(odometry))
        # project back to lidar frame using calib
        lidar_data.transform(lidar_pose.inverse)

        return lidar_data

    def read_pointcloud(
        self,
        frame_id: str,
        lidar_name: str,
        n_sweeps_before: int = 0,
        n_sweeps_after: int = 0,
        motion_compensation: bool = False,
    ) -> LidarData:
        """Read pointcloud from npy format."""
        frame_paths = _get_lidar_frames_path(
            self._frames[frame_id].lidar_frame[lidar_name].filepath,
            n_sweeps_before,
            n_sweeps_after,
        )

        for i, frame_path in enumerate(frame_paths):
            temp_lidar_data = LidarData.from_npy(frame_path)
            if i == 0:
                lidar_data = temp_lidar_data
            else:
                lidar_data.append(temp_lidar_data)

        if motion_compensation:
            lidar_data = self.motion_compensate_pointcloud(lidar_data, frame_id, lidar_name)

        return lidar_data

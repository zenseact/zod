from typing import Optional

import numpy as np

from zod.zod_dataclasses.calibration import LidarCalibration
from zod.zod_dataclasses.oxts import EgoMotion
from zod.zod_dataclasses.sensor import LidarData


def motion_compensate_scanwise(
    lidar_data: LidarData,
    ego_motion: EgoMotion,
    calibration: LidarCalibration,
    target_timestamp: float,
) -> LidarData:
    """Motion compensate a (pointwise compensated) lidar point cloud."""
    lidar_data = lidar_data.copy()
    source_pose = ego_motion.get_poses(lidar_data.core_timestamp)
    target_pose = ego_motion.get_poses(target_timestamp)

    # Compute relative transformation between target pose and source pose
    odometry = np.linalg.inv(target_pose) @ source_pose

    # Project to ego vehicle frame using calib
    lidar_data.transform(calibration.extrinsics)

    # Project to center frame using odometry
    lidar_data.transform(odometry)

    # Project back to lidar frame using calib
    lidar_data.transform(calibration.extrinsics.inverse)
    return lidar_data


def motion_compensate_pointwise(
    lidar_data: LidarData,
    ego_motion: EgoMotion,
    calibration: LidarCalibration,
    target_timestamp: Optional[float] = None,
) -> LidarData:
    """Motion compensate a lidar point cloud in a pointwise manner."""
    lidar_data = lidar_data.copy()
    target_timestamp = target_timestamp or lidar_data.core_timestamp

    # Interpolate oxts data for each frame
    point_poses = ego_motion.get_poses(lidar_data.timestamps)
    target_pose = ego_motion.get_poses(target_timestamp)

    # Compute relative transformation between target pose and point poses
    odometry = np.linalg.inv(target_pose) @ point_poses

    # Project to ego vehicle frame using calib
    lidar_data.transform(calibration.extrinsics)

    # Project to center frame using odometry
    lidar_data.transform(odometry)

    # Project back to lidar frame using calib
    lidar_data.transform(calibration.extrinsics.inverse)
    return lidar_data

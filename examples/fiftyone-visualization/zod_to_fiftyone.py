import os
import sys

import fiftyone as fo
import open3d as o3d
import yaml
from tqdm import tqdm
from utils import normalize_bbox, quaternion_to_euler

import zod.constants as constants
from zod import ZodFrames
from zod.constants import AnnotationProject, Anonymization

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


def create_dataset():
    dataset_root = config["dataset_root"]
    version = config["dataset_version"]
    dataset_name = config["dataset_name"]
    dataset_split = config["dataset_split"]
    pcd_files_dir = config["pcd_files_dir"]
    test_run = config["test_run"]
    mapbox_token = config["mapbox_token"]

    os.makedirs(pcd_files_dir, exist_ok=True)

    # FIFTYONE
    existing_datasets = fo.list_datasets()
    if dataset_name in existing_datasets:
        print(
            f"[FIFTYONE ERROR] Dataset '{dataset_name}' already exists."
            "Delete it, or choose a different name before rerunning."
        )
        sys.exit()

    fo.config.show_progress_bars = False
    dataset = fo.Dataset(name=dataset_name)

    print("Loading ZOD frames...")
    zod_frames = ZodFrames(dataset_root=dataset_root, version=version)

    match dataset_split:
        case "all":
            zod_frames_list = list(zod_frames.get_all_ids())

        case "train":
            zod_frames_list = list(zod_frames.get_split(constants.TRAIN))

        case "val":
            zod_frames_list = list(zod_frames.get_split(constants.VAL))

        case _:
            print("Invalid dataset_split specified.")
            sys.exit()

    if test_run:
        zod_frames_list = zod_frames_list[:10]

    for idx in tqdm(zod_frames_list):
        zod_frame = zod_frames[idx]

        # get image path
        camera_core_frame = zod_frame.info.get_key_camera_frame(Anonymization.BLUR)
        core_image_file = camera_core_frame.filepath

        annotations = zod_frame.get_annotation(AnnotationProject.OBJECT_DETECTION)

        pcd_filename = f"{pcd_files_dir}/{zod_frame.info.id}.pcd"

        if not os.path.exists(pcd_filename):
            core_lidar = zod_frame.get_lidar()[0]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(core_lidar.points)
            o3d.io.write_point_cloud(pcd_filename, pcd)

        # convert ZOD annotations for fiftyone
        detections_3d = []
        detections_2d = []

        for anno in annotations:
            if anno.box3d is not None:
                # 3D boxes
                location = anno.box3d.center
                dimensions = anno.box3d.size

                qw = anno.box3d.orientation[0]
                qx = anno.box3d.orientation[1]
                qy = anno.box3d.orientation[2]
                qz = anno.box3d.orientation[3]
                rotation = quaternion_to_euler(qx, qy, qz, qw)

                detection_3d = fo.Detection(
                    dimensions=list(dimensions),
                    location=list(location),
                    rotation=list(rotation),
                    label=anno.name,
                )

                detections_3d.append(detection_3d)

                # 2D boxes
                detection_2d = fo.Detection(
                    bounding_box=normalize_bbox(anno.box2d.xywh),
                    label=anno.name,
                )

                detections_2d.append(detection_2d)

            else:
                pass

        group = fo.Group()
        samples = [
            fo.Sample(
                filepath=core_image_file,
                group=group.element("image"),
                detections=fo.Detections(detections=detections_2d),
            ),
            fo.Sample(
                filepath=pcd_filename,
                group=group.element("pcd"),
                detections=fo.Detections(detections=detections_3d),
            ),
        ]

        def add_metadata(slice_idx):
            samples[slice_idx]["frame_id"] = zod_frame.metadata.frame_id,
            samples[slice_idx]["time_of_day"] = zod_frame.metadata.time_of_day,
            samples[slice_idx]["country_code"] = zod_frame.metadata.country_code,
            samples[slice_idx]["collection_car"] = zod_frame.metadata.collection_car,
            samples[slice_idx]["road_type"] = zod_frame.metadata.road_type,
            samples[slice_idx]["road_condition"] = zod_frame.metadata.road_condition,
            samples[slice_idx]["num_vehicles"] = zod_frame.metadata.num_vehicles
            samples[slice_idx]["location"] = fo.GeoLocation(
                point=[zod_frame.metadata.longitude, zod_frame.metadata.latitude]
            )

        add_metadata(0) # add metadata to images
        # add_metadata(1) # add metadata to point clouds

        dataset.add_samples(samples)

    # colour by label values by default
    # and change to colour blind friendly colour scheme
    dataset.app_config.color_scheme = fo.ColorScheme(
        color_by="value",
        color_pool=[
            "#E69F00",
            "#56b4e9",
            "#009e74",
            "#f0e442",
            "#0072b2",
            "#d55e00",
            "#cc79a7",
        ],
    )

    # if mapbox token is provided, add it to the app config
    if mapbox_token:
        print("Mapbox token found, enabling map plugin.")
        dataset.app_config.plugins["map"] = {"mapboxAccessToken": mapbox_token}
    else:
        print("Mapbox token not found, map plugin not enabled.")

    dataset.save()

    # keep dataset after session is terminated or not - set in config.yaml
    dataset.persistent = config["dataset_persistent"]


if __name__ == "__main__":
    create_dataset()

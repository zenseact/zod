"""Visualization tools for generating predicted path on images."""

import random
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from dataset.groundtruth_utils import get_ground_truth
from dataset.zod_configs import ZodConfigs
from model.model_config import ModelConfig
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from zod import ZodFrames
from zod.constants import Anonymization, Camera
from zod.data_classes import Calibration
from zod.utils.geometry import (
    get_points_in_camera_fov,
    project_3d_to_2d_kannala,
    transform_points,
)


class ZodImageGenerator:
    """Image generator."""

    def __init__(
        self,
        validation_frames: list,
        zod_frames: ZodFrames,
        n_images: int = 30,
    ) -> None:
        """Generate a set of random images with true and predicted path."""
        self.zod_configs = ZodConfigs()
        self.model_config = ModelConfig()
        self.zod_frames = zod_frames
        self.validation_frames = validation_frames
        self.n_available_samples = len(self.validation_frames)
        self.n_images = n_images
        self._select_image_subset()

    def _select_image_subset(self) -> None:
        """Randomly select <n_images> images from ZOD dataset.

        If a frame is inavlid, generate new random numbers until a valid frame is found.

        """
        self.images = []
        self.frame_ids = []

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    (self.zod_configs.IMG_SIZE, self.zod_configs.IMG_SIZE),
                    antialias=True,
                ),
            ]
        )
        device = torch.device("cuda" if torch.cuda.is_available() and self.model_config.USE_GPU else "cpu")

        while len(self.images) < self.n_images:
            # pick a random frame from dataset
            random_index = random.randint(0, self.n_available_samples - 1)  # noqa: S311
            frame_id = self.validation_frames[random_index]

            try:
                zod_frame = self.zod_frames[frame_id]

                image = zod_frame.get_image(Anonymization.DNAT)
                image = transform(image)
                image = image.to(device)
                image = image.unsqueeze(0)

                # if prediction is valid, ad frame to image subset
                self.images.append(image)
                self.frame_ids.append(frame_id)

            # if frame invalid
            except (TypeError, ValueError, FileNotFoundError):
                pass

    def visualize_prediction_on_image(self, model: torch.nn.Module, save_tag: str) -> torch.Tensor:
        """Visualize true and predicted holistic path on each image.

        Also combine the images into a grid.

        Args:
            model (torch.nn.Module): Model
            save_tag (str): String to tag the saved images

        Returns:
            torch.Tensor: grid image with visualized predictions

        """
        device = torch.device("cuda" if torch.cuda.is_available() and self.model_config.USE_GPU else "cpu")
        model.eval()
        model.to(device)

        transform = transforms.Compose([transforms.ToTensor()])
        image_tensors = []

        for frame_id, image in zip(self.frame_ids, self.images):
            try:
                with torch.no_grad():
                    predicted_path = model(image)[0, :]
                zod_frame = self.zod_frames[frame_id]
                raw_image = zod_frame.get_image(Anonymization.DNAT)
                trajectories = list(predicted_path[: -self.model_config.NR_OF_MODES].reshape((-1, 51)).cpu().numpy())
                mode_probabilities = list(predicted_path[-self.model_config.NR_OF_MODES :])

                final_image = self._visualize_paths_on_image(
                    raw_image,
                    self.zod_frames,
                    frame_id,
                    trajectories,
                    mode_probabilities,
                )

                image_tensor = transform(final_image)
                image_tensors.append(image_tensor)

            # if frame invalid
            except (TypeError, ValueError) as e:
                print(f"skipping frame {frame_id} with error: {e}")
                pass

        stacked_tensors = torch.stack(image_tensors)
        grid = make_grid(stacked_tensors, nrow=3, padding=5)

        transform = transforms.Compose(
            [
                transforms.Resize(
                    (
                        round(self.zod_configs.IMG_SIZE * (self.n_images // 3)),
                        round(self.zod_configs.IMG_SIZE * 3),
                    ),
                    antialias=True,
                ),
            ]
        )

        grid = transform(grid)

        grid.cpu().detach()
        model.train()

        # save to results/
        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)
        out_path = results_dir / f"predictions_grid_{save_tag}.png"
        save_image(grid, out_path)

    def _draw_line(self, image: np.array, line: np.array, color: tuple) -> np.array:
        """Draw points on image.

        Args:
            image (np.array): image
            line (np.array): points
            color (tuple): color RGB tuple

        Returns:
            image(np.array): image with line

        """
        return cv2.polylines(
            image.copy(),
            [np.round(line).astype(np.int32)],
            isClosed=False,
            color=color,
            thickness=60,
        )

    def _transform_absolute_to_relative_path(
        self,
        image: np.array,
        points: np.array,
        camera: str,
        calibrations: Calibration,
    ) -> np.array:
        t_inv = np.linalg.pinv(calibrations.get_extrinsics(camera).transform)
        points = points.reshape(((self.zod_configs.NUM_OUTPUT // 3), 3))
        camerapoints = transform_points(points[:, :3], t_inv)

        # filter points that are not in the camera field of view
        points_in_fov = get_points_in_camera_fov(calibrations.cameras[camera].field_of_view, camerapoints)
        points_in_fov = points_in_fov[0]

        # project points to image plane
        xy_array = project_3d_to_2d_kannala(
            points_in_fov,
            calibrations.cameras[camera].intrinsics[..., :3],
            calibrations.cameras[camera].distortion,
        )

        points = []
        for i in range(xy_array.shape[0]):
            x, y = int(xy_array[i, 0]), int(xy_array[i, 1])
            cv2.circle(image, (x, y), 2, (255, 0, 0), -1)
            points.append([x, y])

        return points, image

    def _visualize_paths_on_image(
        self,
        image: np.ndarray,
        zod_frames: ZodFrames,
        frame_id: int,
        predicted_paths: List = None,
        probabilities: List = None,
    ) -> np.array:
        """Visualize oxts track on image plane."""
        camera = Camera.FRONT
        zod_frame = zod_frames[frame_id]
        calibrations = zod_frame.calibration
        true_path = get_ground_truth(zod_frames, frame_id, self.zod_configs)

        # add true path to image
        true_path_pov, image = self._transform_absolute_to_relative_path(image, true_path, camera, calibrations)

        ground_truth_color = (20, 150, 61)  # (19, 80, 41)
        image = self._draw_line(image, true_path_pov, ground_truth_color)

        # add predicted path to image
        if predicted_paths is None:
            return image
        mid_points = []
        predictions_color = (161, 65, 137)
        for predicted_path in predicted_paths:
            # transform point to camera coordinate system
            predicted_path_pov, image = self._transform_absolute_to_relative_path(
                image, predicted_path, camera, calibrations
            )

            image = self._draw_line(image, predicted_path_pov, predictions_color)
            mid_points.append(self._get_mid_points(predicted_path_pov))
        if probabilities is not None:
            for i, probability in enumerate(probabilities):
                image = self._add_probability_text(str(int(round(probability.item() * 100))), image, mid_points[i])
        return image

    def _get_mid_points(self, points: List) -> tuple:
        return (
            points[len(points) // 2][0],
            points[len(points) // 2][1],
        )

    def _add_probability_text(self, prob: str, image: np.ndarray, point: tuple) -> np.ndarray:
        """Adds the prob string on the image at the point."""
        cv2.putText(
            image,
            prob + "%",
            point,
            cv2.FONT_HERSHEY_SIMPLEX,
            5,
            (255, 255, 255),
            10,
        )
        return image

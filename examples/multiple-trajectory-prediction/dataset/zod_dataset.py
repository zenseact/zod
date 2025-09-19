"""Zod dataset, dataset class."""

import cv2
from dataset.groundtruth_utils import get_ground_truth
from dataset.zod_configs import ZodConfigs
from torch.utils.data import Dataset
from torchvision import transforms

from zod import ZodFrames
from zod.constants import Anonymization, Camera


class ZodDataset(Dataset):
    """Zod dataset class."""

    def __init__(
        self,
        zod_frames: list,
        frames_id_set: list,
        stored_ground_truth: dict = None,
        transform: transforms = None,
        zod_configs: ZodConfigs = None,
    ) -> None:
        self.zod_frames: ZodFrames = zod_frames
        self.frames_id_set = frames_id_set
        self.transform = transform if transform is not None else transforms.ToTensor()
        self.stored_ground_truth = stored_ground_truth
        self.zod_configs = zod_configs

    def __len__(self) -> int:
        """Get number of frames."""
        return len(self.frames_id_set)

    def __getitem__(self, idx: int) -> tuple:
        """Iterator."""
        frame_idx = self.frames_id_set[idx]
        frame = self.zod_frames[frame_idx]

        if self.zod_configs.USE_PRE_RESIZED_IMGS:
            original_path = frame.info.get_key_camera_frame(
                camera=Camera.FRONT, anonymization=Anonymization.DNAT
            ).filepath

            resized_image_path = original_path.rsplit(".", 1)[0] + "_resized.jpg"
            try:
                image = cv2.imread(resized_image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except (FileNotFoundError, TypeError):
                image = None
                print(f"Image {resized_image_path} not found.")

        else:
            image = frame.get_image(Anonymization.DNAT)
        label = None

        label = (
            self.stored_ground_truth[frame_idx]
            if self.stored_ground_truth
            else get_ground_truth(self.zod_frames, frame_idx, self.zod_configs)
        )

        label = label.astype("float32")

        if self.transform:
            image = self.transform(image)
        return image, label

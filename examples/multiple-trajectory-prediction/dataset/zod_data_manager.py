"""Zod dataset manager."""

from typing import Tuple

from dataset.groundtruth_utils import load_ground_truth
from dataset.zod_configs import ZodConfigs
from dataset.zod_dataset import ZodDataset
from dataset.zod_image_generator import ZodImageGenerator
from torch import Generator
from torch.utils.data import DataLoader, RandomSampler, random_split
from torchvision import transforms

from zod import ZodFrames, constants


class ZodDatasetManager:
    """Zod dataset manager class."""

    def __init__(self) -> None:
        self.zod_configs = ZodConfigs()
        self.zod_frames = ZodFrames(dataset_root=self.zod_configs.DATASET_ROOT, version="full")
        self.transform = self._get_transform()
        self.ground_truth = load_ground_truth(self.zod_configs.STORED_GROUND_TRUTH_PATH)
        self.test_frames = None

    def get_test_dataloader(self) -> DataLoader:
        """Load the ZOD test dataset from the VAL partition. Server side test set."""
        validation_frames_all = self.zod_frames.get_split(constants.VAL)

        validation_frames_all = [idx for idx in validation_frames_all if self._is_valid_frame(idx, self.ground_truth)]

        validation_frames = validation_frames_all[: int(len(validation_frames_all) * self.zod_configs.TEST_SIZE)]

        self.test_frames = validation_frames

        testset = ZodDataset(
            zod_frames=self.zod_frames,
            frames_id_set=validation_frames,
            stored_ground_truth=self.ground_truth,
            transform=self.transform,
            zod_configs=self.zod_configs,
        )
        print(f"Test dataset loaded. Length: {len(testset)}")
        return DataLoader(testset, batch_size=self.zod_configs.BATCH_SIZE)

    def get_train_val_dataloader(self, seed: int = 42) -> Tuple[DataLoader, DataLoader]:
        """Get train and validation dataloader for client side."""
        train_frames = self.zod_frames.get_split(constants.TRAIN)

        train_frames = [idx for idx in train_frames if self._is_valid_frame(idx, self.ground_truth)]
        trainset = ZodDataset(
            zod_frames=self.zod_frames,
            frames_id_set=train_frames,
            stored_ground_truth=self.ground_truth,
            transform=self.transform,
            zod_configs=self.zod_configs,
        )

        # Split into train/val and create DataLoader
        len_val = int(len(trainset) * self.zod_configs.VAL_SIZE)
        len_train = int(len(trainset) - len_val)

        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(trainset, lengths, Generator().manual_seed(seed))
        train_sampler = RandomSampler(ds_train)
        trainloader = DataLoader(
            ds_train,
            batch_size=self.zod_configs.BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            sampler=train_sampler,
        )
        valloader = DataLoader(ds_val, batch_size=self.zod_configs.BATCH_SIZE, num_workers=0)

        return trainloader, valloader

    def get_image_generator(self) -> ZodImageGenerator:
        """Get image generator for ZOD hollistic path."""
        if self.test_frames is None:
            self.get_test_dataloader()
        return ZodImageGenerator(self.test_frames, self.zod_frames)

    def _get_transform(self) -> transforms.Compose:
        """Get transform to use."""
        return (
            transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(self.zod_configs.NORMALIZE_MEAN, self.zod_configs.NORMALIZE_STD),
                ]
            )
            if self.zod_configs.USE_PRE_RESIZED_IMGS
            else transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(
                        (self.zod_configs.IMG_SIZE, self.zod_configs.IMG_SIZE),
                        antialias=True,
                    ),
                    transforms.Normalize(self.zod_configs.NORMALIZE_MEAN, self.zod_configs.NORMALIZE_STD),
                ]
            )
        )

    def _is_valid_frame(self, frame_id: str, ground_truth: dict) -> bool:
        """Check if frame is valid."""
        if frame_id == "005350":
            return False

        return frame_id in ground_truth

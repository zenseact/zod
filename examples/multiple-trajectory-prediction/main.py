"""Main script."""

from dataset.zod_data_manager import ZodDatasetManager
from model.multi_trajectory_model_manager import MultiTrajectoryModelManager


def main() -> None:
    """Run experiment."""
    # get data
    data_manager = ZodDatasetManager()
    test_loader = data_manager.get_test_dataloader()
    train_loader, val_loader = data_manager.get_train_val_dataloader()
    image_generator = data_manager.get_image_generator()

    # create model and train
    model_manager = MultiTrajectoryModelManager()
    model_manager.train(train_loader, val_loader, image_generator)

    # test
    model_manager.test(test_loader)


if __name__ == "__main__":
    main()

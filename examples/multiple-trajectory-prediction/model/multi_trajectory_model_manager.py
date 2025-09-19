"""Model manager class for Holistic Path model trained with ZOD."""

import time
from typing import Tuple

import numpy as np
import torch
from dataset.zod_image_generator import ZodImageGenerator
from model.model import Net
from model.model_config import ModelConfig
from torch.utils.data import DataLoader


class MultiTrajectoryModelManager:
    """Zod dataset manager class."""

    def __init__(self) -> None:
        self.model_configs = ModelConfig()
        device = torch.device("cuda" if torch.cuda.is_available() and self.model_configs.USE_GPU else "cpu")
        self.net = Net(self.model_configs).to(device)

    def train(
        self, trainloader: DataLoader, valloader: DataLoader, image_generator: ZodImageGenerator
    ) -> Tuple[list, list]:
        """Trains data to update the model parameters.

        Args:
            trainloader (torch.utils.data.DataLoader): train loader
            valloader (torch.utils.data.DataLoader): validaiton loader
            image_generator (ZodImageGenerator): Image generator to visualize results

        Returns:
            tuple[list, list]: epoch_train_losses, epoch_val_losses

        """
        device = torch.device("cuda" if torch.cuda.is_available() and self.model_configs.USE_GPU else "cpu")
        epochs = self.model_configs.EPOCHS
        self.net.train()
        opt = torch.optim.Adam(self.net.parameters(), lr=self.model_configs.LEARNING_RATE)
        epoch_train_losses = []
        epoch_val_losses = []
        for epoch in range(1, epochs + 1):
            tstart = time.time()
            batch_train_losses = []
            for data_, target_ in trainloader:
                data, target = data_.to(device), target_.to(device)
                opt.zero_grad()
                output = self.net(data)
                loss = self.net.loss_fn(output, target)
                loss.backward()
                opt.step()
                batch_train_losses.append(loss.item())
            epoch_train_losses.append(sum(batch_train_losses) / len(batch_train_losses))
            val_loss, _ = self.test(valloader)
            epoch_val_losses.append(val_loss)
            print(
                f"Epoch completed in {time.time() - tstart:.2f} seconds with "
                + f"{len(trainloader)} batches of batch size {trainloader.batch_size}"
            )
            print(f"Train loss for epoch {epoch}: {epoch_train_losses[-1]:.2f}")
            print(f"Validation loss for epoch {epoch}: {epoch_val_losses[-1]:.2f}")
            image_generator.visualize_prediction_on_image(self.net, f"epoch_{str(epoch)}")
        return epoch_train_losses, epoch_val_losses

    def test(self, testloader: DataLoader) -> Tuple[float, float]:
        """Test the model performance.

        Returns a Tuple with [loss, accuracy], let accuracy be None on regression tasks.
        """
        device = torch.device("cuda" if torch.cuda.is_available() and self.model_configs.USE_GPU else "cpu")
        criterion = self.net.loss_fn
        self.net.eval()
        loss = []
        with torch.no_grad():
            for images_, labels_ in testloader:
                images, labels = images_.to(device), labels_.to(device)
                outputs = self.net(images)
                loss.append(criterion(outputs, labels).item())
        self.net.train()
        return np.mean(loss), None

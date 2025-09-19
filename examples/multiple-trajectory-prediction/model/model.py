"""Models."""

from typing import Tuple

import pytorch_lightning as pl
import torch
from model.model_config import ModelConfig
from torch import nn
from torch.nn import functional as f
from torchvision import models


class MultiTrajectoryLoss:
    """Computes MultiTrajectoryLoss."""

    def __init__(self, num_modes: int) -> None:
        self.num_modes = num_modes

    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes the MultiTrajectoryLoss loss on a batch."""
        batch_losses = torch.Tensor().requires_grad_(True).to(predictions.device)
        trajectories, modes = self._get_trajectory_and_modes(predictions)
        for sample_idx in range(predictions.shape[0]):
            best_mode = self._compute_best_mode(target=targets[sample_idx], trajectories=trajectories[sample_idx])
            best_mode_trajectory = trajectories[sample_idx, best_mode].reshape(-1)
            regression_loss = f.smooth_l1_loss(best_mode_trajectory, targets[sample_idx])
            mode_probabilities = modes[sample_idx].unsqueeze(0)
            best_mode_target = torch.tensor([best_mode], device=predictions.device)
            classification_loss = f.cross_entropy(mode_probabilities, best_mode_target)
            loss = classification_loss + regression_loss
            batch_losses = torch.cat((batch_losses, loss.unsqueeze(0)), 0)
        return torch.mean(batch_losses)

    def _compute_best_mode(self, target: torch.tensor, trajectories: torch.tensor) -> torch.tensor:
        """Finds the index of the best mode based on l1 norm from the ground truth."""
        l1_norms = torch.empty(trajectories.shape[0])

        for i in range(trajectories.shape[0]):
            l1_norm = torch.sum(torch.abs(trajectories[0, i] - target))
            l1_norms[i] = l1_norm

        return torch.argmin(l1_norms)

    def _get_trajectory_and_modes(self, model_prediction: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Splits the predictions from the model into mode probabilities and trajectory."""
        mode_probabilities = model_prediction[:, -self.num_modes :].clone()

        desired_shape = (
            model_prediction.shape[0],
            self.num_modes,
            -1,
        )
        trajectories = model_prediction[:, : -self.num_modes].clone().reshape(desired_shape)

        return trajectories, mode_probabilities


class Net(pl.LightningModule):
    """Neural CNN model class."""

    def __init__(self, model_configs: ModelConfig) -> None:
        super(Net, self).__init__()
        self.model_configs = model_configs
        self.model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)

        self.loss_fn = MultiTrajectoryLoss(self.model_configs.NR_OF_MODES)

        device = torch.device("cuda" if torch.cuda.is_available() and self.model_configs.USE_GPU else "cpu")

        self.target_dists = torch.Tensor(self.model_configs.TARGET_DISTANCES).to(device)
        self.num_target_distances = len(self.model_configs.TARGET_DISTANCES)

        self.num_modes = self.model_configs.NR_OF_MODES

        self._change_head_net(self.num_target_distances * 3, self.num_modes)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Forward propagation."""
        model_output = self.model(image)

        trajectories = model_output[:, : -self.num_modes].reshape((-1, self.num_target_distances, 3))
        mode_probabilities = model_output[:, -self.num_modes :]
        scaling_factors = self.target_dists.view(-1, 1)
        trajectories *= scaling_factors

        if not self.training:
            mode_probabilities = f.softmax(mode_probabilities, dim=-1)

        trajectories_reshaped = trajectories.reshape((-1, 3 * self.num_modes * self.num_target_distances))

        return torch.cat([trajectories_reshaped, mode_probabilities], dim=-1)

    def _change_head_net(self, num_points: int, num_modes: int) -> None:
        """Change the last model classifier step."""
        num_ftrs = self.model.classifier[-1].in_features
        head_net = nn.Sequential(
            nn.Linear(num_ftrs, 1024, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(
                512,
                num_points * num_modes + num_modes,
                bias=True,
            ),
        )

        self.model.classifier[-1] = head_net

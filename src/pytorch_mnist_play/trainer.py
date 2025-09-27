from pydantic import BaseModel, ConfigDict, ValidationError
import logging

import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from typing import Sized


class Config(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    batch_size: int = 128
    epochs: int = 20
    lr: float = 0.01
    momentum: float = 0.9
    seed: int = 1
    log_dir: str = "logs/mnist"
    summaryWriter: SummaryWriter


class Trainer:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.summaryWriter = config.summaryWriter

    def _optimizer(self, model: nn.Module, **kwargs) -> optim.Optimizer:
        optimizer = optim.SGD(
            model.parameters(),
            lr=self.config.lr,
            momentum=self.config.momentum,
            **kwargs,
        )
        self.scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        return optimizer

    def train_one_epoch(
        self,
        model: nn.Module,
        device: torch.device,
        train_loader: torch.utils.data.DataLoader,
        optimizer: optim.Optimizer,
        epoch: int,
    ):
        model.train()
        pbar = tqdm(train_loader, leave=True, desc=f"Epoch {epoch}")

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)

            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=f"{loss.item():.4f}", batch_id=batch_idx)
        pbar.close()
        # self.scheduler.step()

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        device: torch.device,
        test_loader: torch.utils.data.DataLoader,
        epoch: int,
        model_name: str,
    ):
        assert isinstance(test_loader.dataset, Sized), "Dataset must implement __len__"
        model.eval()

        dataset_size = len(test_loader.dataset)
        test_loss_cumulative = 0.0
        correct = 0

        for data, target in test_loader:
            # data, target = data.to(device), target.to(device)
            data: torch.Tensor = data.to(device)
            target: torch.Tensor = target.to(device)

            output: torch.Tensor = model(data)

            test_loss_cumulative += F.nll_loss(output, target, reduction="sum").item()
            pred: torch.Tensor = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss_average = test_loss_cumulative / dataset_size
        accuracy = 100.0 * correct / dataset_size

        self.summaryWriter.add_scalars(
            "Metrics/Loss", {model_name: test_loss_average}, global_step=epoch
        )
        self.summaryWriter.add_scalars(
            "Metrics/Accuracy", {model_name: accuracy}, global_step=epoch
        )
        self.summaryWriter.flush()

        print(
            f"[{model_name}] Epoch {epoch:02d} - Loss: {test_loss_average:.4f}, Accuracy: {correct}/{dataset_size} ({accuracy:.2f}%)"
        )
        return test_loss_average, accuracy

    def fit(
        self,
        model: nn.Module,
        device: torch.device,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        model_name: str,
    ):
        optimizer = self._optimizer(model)
        try:
            for epoch in range(1, self.config.epochs + 1):
                self.train_one_epoch(model, device, train_loader, optimizer, epoch)
                self.evaluate(model, device, test_loader, epoch, model_name)
        except Exception as e:
            print(f"An unexpected error occurred during fit: {type(e).__name__}: {e}")

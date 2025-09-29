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
        # Tensorboard writer
        self.summaryWriter = config.summaryWriter
        # Logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

    def _log_metrics(self, metrics: dict, epoch: int, model_name: str):
        for key, value in metrics.items():
            self.summaryWriter.add_scalars(
                f"Metrics/{key}", {model_name: value}, global_step=epoch
            )
        self.summaryWriter.flush()

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
        model_name: str,
    ):
        model.train()
        pbar = tqdm(train_loader, leave=True, desc=f"Epoch {epoch}")

        assert isinstance(train_loader.dataset, Sized), "Dataset must implement __len__"
        dataset_size = len(train_loader.dataset)
        correct = 0

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)

            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            pred: torch.Tensor = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            pbar.set_postfix(loss=f"{loss.item():.4f}", batch_id=batch_idx)
        pbar.close()
        # self.scheduler.step()

        accuracy = 100.0 * correct / dataset_size
        self._log_metrics(
            {"Train Accuracy": accuracy},
            epoch,
            model_name,
        )
        self.logger.info(
            f"[TRAIN {model_name}] Epoch {epoch:02d} - Accuracy: {correct}/{dataset_size} ({accuracy:.2f}%)"
        )

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

        self._log_metrics(
            {"Test Loss": test_loss_average, "Test Accuracy": accuracy},
            epoch,
            model_name,
        )

        self.logger.info(
            f"[TEST {model_name}] Epoch {epoch:02d} - Loss: {test_loss_average:.4f}, Accuracy: {correct}/{dataset_size} ({accuracy:.2f}%)"
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
                self.train_one_epoch(
                    model, device, train_loader, optimizer, epoch, model_name
                )
                self.evaluate(model, device, test_loader, epoch, model_name)
        except RuntimeError as e:
            self.logger.error(f"RuntimeError during fit: {e}")
        except ValueError as e:
            self.logger.error(f"ValueError during fit: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error during fit: {type(e).__name__}: {e}")

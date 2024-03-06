import torch
import torch.nn as nn
from torch import Tensor


class StochLoss(nn.Module):
    def __init__(
        self,
        dim: int = 1,
        increase_type: str = "exponential",
    ):
        super(StochLoss, self).__init__()
        self.dim: int = dim
        self.increase_type: str = increase_type
        self.func: callable = (
            self._linear if increase_type == "linear" else self._exponential
        )

    def _linear(self, y_hat: Tensor) -> Tensor:
        return torch.linspace(1, 10, steps=y_hat.size(self.dim))

    def _exponential(self, y_hat: Tensor) -> Tensor:
        return torch.exp(self._linear(y_hat))

    def forward(self, y: Tensor, y_hat: Tensor) -> Tensor:
        weights = self.func(y_hat)
        errors = y - y_hat
        errors = errors.mT * weights
        errors **= 2
        loss = errors.mean()
        return loss

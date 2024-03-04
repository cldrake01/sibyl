import torch
import torch.nn as nn
from torch import Tensor


class StochLoss(nn.Module):
    def __init__(
        self,
        dim: int = 1,
        increase_type: str = "exponential",
        max_weight: int | float = 1,
        alpha: float = 0.99,
    ):
        super(StochLoss, self).__init__()
        self.dim: int = dim
        self.increase_type: str = increase_type
        self.max_weight: int | float = max_weight
        self.func: callable = (
            self._linear if increase_type == "linear" else self._exponential
        )

    def _linear(self, target: Tensor) -> Tensor:
        return torch.linspace(1, self.max_weight, steps=target.size(self.dim))

    def _exponential(self, target: Tensor) -> Tensor:
        return torch.exp(
            torch.linspace(0, target.size(self.dim), steps=target.size(self.dim))
        )

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        weights = self.func(target)
        squared_errors = torch.div(
            torch.abs(torch.sum(input - target, dim=self.dim)), target.size(self.dim)
        )
        weighted_squared_errors = squared_errors.mT * weights
        loss = weighted_squared_errors.mean()
        return loss

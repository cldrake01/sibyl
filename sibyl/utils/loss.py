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

    def _linear(self, target: Tensor) -> Tensor:
        return torch.linspace(1, 10, steps=target.size(self.dim))

    def _exponential(self, target: Tensor) -> Tensor:
        return torch.exp(self._linear(target))

    def forward(self, input: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        mse = torch.nn.functional.mse_loss(input, target)
        weights = self.func(target)
        errors = input - target
        # We divide because our steps are between 0 and 1.
        errors = errors.mT * weights
        errors **= 2
        loss = errors.mean()
        return loss, mse

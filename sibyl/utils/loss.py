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
        # weights = self.func(y_hat)
        # errors = y - y_hat
        # errors = errors.mT * weights
        # errors **= 2
        # loss = errors.mean()
        # [1, 15, 4] -> [15, 4]
        y, y_hat = y.squeeze(), y_hat.squeeze()
        # We multiply by 100 because covariance matrices contain values between -1 and 1
        y_cov, y_hat_cov = y.cov() * 100, y_hat.cov() * 100
        y, y_hat = y * y_cov, y_hat * y_hat_cov
        print(y)
        print(y.size())
        print(y_hat)
        print(y_hat.size())
        errors = y - y_hat
        errors **= 2
        loss = errors.mean()
        return loss

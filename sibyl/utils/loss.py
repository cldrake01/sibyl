import torch
import torch.nn as nn
from torch import Tensor


class StochLoss(nn.Module):
    def __init__(
            self,
            dim: int = 1,
    ):
        super(StochLoss, self).__init__()
        self.dim: int = dim

    def _weights(self, y_hat: Tensor) -> Tensor:
        n = y_hat.size(self.dim)
        l = torch.linspace(1, n, n).int()
        return torch.exp(
            torch.cov(
                y_hat.squeeze().mT,
                fweights=l
            )
        )

    def forward(self, y: Tensor, y_hat: Tensor) -> Tensor:
        weights = self._weights(y_hat)
        print(weights.size())
        errors = y - y_hat
        print(errors.size())
        errors = errors * weights
        errors **= 2
        loss = errors.mean()
        return loss

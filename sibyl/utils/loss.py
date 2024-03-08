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

    def _weights(self, t: Tensor) -> Tensor:
        n = t.size(self.dim)
        l = torch.linspace(1, n, n  ).int()
        w = torch.repeat_interleave(t, l, dim=self.dim)
        return w

    def forward(self, y: Tensor, y_hat: Tensor) -> Tensor:
        errors = self._weights(y - y_hat)**2
        loss = errors.var()
        return loss

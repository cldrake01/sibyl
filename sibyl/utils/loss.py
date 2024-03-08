import pickle

import torch
import torch.nn as nn
from torch import Tensor


class VarianceLoss(nn.Module):
    def __init__(
        self,
        dim: int = 1,
    ):
        super(VarianceLoss, self).__init__()
        self.dim: int = dim
        self.loss: list = []

    def _weights(self, t: Tensor) -> Tensor:
        n = t.size(self.dim)
        l = torch.linspace(1, n, n).int()
        w = torch.repeat_interleave(t, l, dim=self.dim)
        return w

    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:
        errors: Tensor = self._weights(y - y_hat) ** 2
        loss = errors.var()
        self.loss.append(torch.nn.functional.mse_loss(y, y_hat).item())
        with open("vl.pkl", "wb") as f:
            pickle.dump(self.loss, f)
        return loss


class EigenLoss(nn.Module):
    def __init__(self, dim: int = 1, *args, **kwargs):
        super(EigenLoss, self).__init__()
        self.dim: int = dim

    def _weights(self, t: Tensor) -> Tensor:
        return 0

    def forward(self, y: Tensor, y_hat: Tensor) -> Tensor:
        return 0

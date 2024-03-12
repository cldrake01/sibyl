import pickle

import torch
import torch.nn as nn
from torch import Tensor


class VarianceLoss(nn.Module):
    def __init__(
        self,
        dim: int = 1,
        weighted: bool = True,
        benchmark: bool = False,
    ):
        super(VarianceLoss, self).__init__()
        self.dim: int = dim
        self.variance_loss: list = []
        self.sum_loss: list = []
        self.func: callable = self._sum_loss if benchmark else self._variance_loss
        self.weights: callable = self._weights if weighted else lambda x: x
        # self.file: str = (
        #     ("w" if weighted else "") + ("s" if benchmark else "v") + ".pkl"
        # )

    def _weights(self, t: Tensor) -> Tensor:
        n = t.size(self.dim)
        l = torch.linspace(1, n, n).int()
        w = torch.repeat_interleave(t, l, dim=self.dim)
        return w

    def _variance_loss(self, y: Tensor, y_hat: Tensor) -> Tensor:
        errors = (y - y_hat).abs()
        # self.variance_loss.append(errors.sum().item())
        # pickle.dump(self.variance_loss, open(self.file, "wb"))
        errors = self._weights(errors)
        return errors.std()

    def _sum_loss(self, y: Tensor, y_hat: Tensor) -> Tensor:
        errors = (y - y_hat) ** 2
        # self.sum_loss.append(errors.abs().sum().item())
        # pickle.dump(self.sum_loss, open(self.file, "wb"))
        return errors.mean()

    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:
        return self.func(y, y_hat)


class EigenLoss(nn.Module):
    def __init__(self, dim: int = 1, *args, **kwargs):
        super(EigenLoss, self).__init__()
        self.dim: int = dim

    def _weights(self, t: Tensor) -> Tensor:
        return 0

    def forward(self, y: Tensor, y_hat: Tensor) -> Tensor:
        return 0

import torch
import torch.nn as nn
from torch import Tensor


class MaxAE(nn.Module):
    def __init__(
        self,
        dim: int = 1,
        weighted: bool = True,
        benchmark: bool = False,
    ):
        super(MaxAE, self).__init__()
        self.dim: int = dim
        self.loss: callable = self._mae if benchmark else self._maxae
        self.weights: callable = self._weights if weighted else lambda x: x

    def __call__(self, *args, **kwargs):
        return self._maxae(*args, **kwargs)

    def _weights(self, t: Tensor) -> Tensor:
        n = t.size(self.dim)
        l = torch.linspace(1, n, n).int()
        t = torch.repeat_interleave(t, l, dim=self.dim)
        t = torch.exp(t)
        return t

    def _maxae(self, y: Tensor, y_hat: Tensor) -> Tensor:
        r = (y - y_hat).abs()
        w = torch.exp(
            torch.abs(torch.var(y, dim=self.dim) - torch.var(y_hat, dim=self.dim))
        )
        return w.sum() * r.sum()

    @staticmethod
    def _mae(y: Tensor, y_hat: Tensor) -> Tensor:
        return torch.nn.functional.l1_loss(y, y_hat)

    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:
        loss = self.loss(y, y_hat)
        return loss


class MaxSE(nn.Module):
    def __init__(
        self,
        dim: int = 1,
        weighted: bool = True,
        benchmark: bool = False,
    ):
        super(MaxSE, self).__init__()
        self.dim: int = dim
        self.loss: callable = self._mse if benchmark else self._maxse
        self.weights: callable = self._weights if weighted else lambda x: x

    def __call__(self, *args, **kwargs):
        return self._maxse(*args, **kwargs)

    def _weights(self, t: Tensor) -> Tensor:
        n = t.size(self.dim)
        l = torch.linspace(1, n, n).int()
        t = torch.repeat_interleave(t, l, dim=self.dim)
        t = torch.exp(t)
        return t

    def _maxse(self, y: Tensor, y_hat: Tensor) -> Tensor:
        r = (y - y_hat) ** 2
        w = torch.exp(
            torch.abs(torch.var(y, dim=self.dim) - torch.var(y_hat, dim=self.dim))
        )
        return w.max() * r.max()

    @staticmethod
    def _mse(y: Tensor, y_hat: Tensor) -> Tensor:
        return torch.nn.functional.mse_loss(y, y_hat)

    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:
        return self.loss(y, y_hat)

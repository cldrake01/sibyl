import torch
import torch.nn as nn
from torch import Tensor


class MaxAE(nn.Module):
    def __init__(
        self,
        dim: int = 1,
        weighted: bool = True,
        benchmark: bool = True,
    ):
        super(MaxAE, self).__init__()
        self.dim: int = dim
        self.variance_loss: list = []
        self.sum_loss: list = []
        self.func: callable = self._mae if benchmark else self._maxae
        self.weights: callable = self._weights if weighted else lambda x: x
        # self.file: str = (
        #     ("w" if weighted else "") + ("s" if benchmark else "v") + ".pkl"
        # )

    def _weights(self, t: Tensor) -> Tensor:
        n = t.size(self.dim)
        l = torch.linspace(1, n, n).int()
        w = torch.repeat_interleave(t, l, dim=self.dim)
        print(w)
        w = torch.exp(l)
        return w

    def _maxae(self, y: Tensor, y_hat: Tensor) -> Tensor:
        r = (y - y_hat).abs()
        w = torch.exp(
            torch.abs(torch.var(y, dim=self.dim) - torch.var(y_hat, dim=self.dim))
        )
        return w.max() * r.max()

    def _mae(self, y: Tensor, y_hat: Tensor) -> Tensor:
        return torch.nn.functional.l1_loss(y, y_hat)

    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:
        return self.func(y, y_hat)


class MaxSE(nn.Module):
    def __init__(
        self,
        dim: int = 1,
        weighted: bool = True,
        benchmark: bool = True,
    ):
        super(MaxSE, self).__init__()
        self.dim: int = dim
        self.variance_loss: list = []
        self.sum_loss: list = []
        self.func: callable = self._mse if benchmark else self._maxse
        self.weights: callable = self._weights if weighted else lambda x: x
        # self.file: str = (
        #     ("w" if weighted else "") + ("s" if benchmark else "v") + ".pkl"
        # )

    def _weights(self, t: Tensor) -> Tensor:
        n = t.size(self.dim)
        l = torch.linspace(1, n, n).int()
        w = torch.repeat_interleave(t, l, dim=self.dim)
        print(w)
        w = torch.exp(l)
        return w

    def _maxse(self, y: Tensor, y_hat: Tensor) -> Tensor:
        r = (y - y_hat) ** 2
        w = torch.exp(
            torch.abs(torch.var(y, dim=self.dim) - torch.var(y_hat, dim=self.dim))
        )
        return w.max() * r.max()

    def _mse(self, y: Tensor, y_hat: Tensor) -> Tensor:
        return torch.nn.functional.mse_loss(y, y_hat)

    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:
        return self.func(y, y_hat)

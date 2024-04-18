import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Fourier(nn.Module):
    def __init__(
        self,
        dim: int = 1,
    ):
        """
        A fourier loss function.

        :param dim: The dimension along which to compute the variance.
        """
        super(Fourier, self).__init__()
        self._dim: int = dim

    def __call__(self, *args, **kwargs):
        return self.fourier(*args, **kwargs)

    @staticmethod
    def fourier(y: Tensor, y_hat: Tensor) -> Tensor:
        y = (y - y.mean()) / y.max()
        y_hat = (y_hat - y_hat.mean()) / y_hat.max()
        i = torch.trapezoid((y - y_hat).abs()).sum()
        return i

    @staticmethod
    def mae(y: Tensor, y_hat: Tensor) -> float:
        return F.l1_loss(y, y_hat).item()

    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:
        return self.fourier(y, y_hat)


class VMaxAE(nn.Module):
    def __init__(
        self,
        dim: int = 1,
    ):
        """
        Variance-weighted Maximum Absolute Error (VMaxAE) loss function.

        :param dim: The dimension along which to compute the variance.
        """
        super(VMaxAE, self).__init__()
        self._dim: int = dim

    def __call__(self, *args, **kwargs):
        return self.vmaxae(*args, **kwargs)

    def _weight(self, x: Tensor) -> Tensor:
        return torch.repeat_interleave(
            x,
            torch.tensor(range(1, x.size(dim=self._dim) + 1)),
            dim=self._dim,
        )

    def vmaxae(self, y: Tensor, y_hat: Tensor) -> Tensor:
        r = (y - y_hat).abs()
        w = torch.abs(torch.var(y, dim=self._dim) - torch.var(y_hat, dim=self._dim)) + 1
        return r.max() * w.max()

    @staticmethod
    def mae(y: Tensor, y_hat: Tensor) -> float:
        return F.l1_loss(y, y_hat).item()

    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:
        return self.vmaxae(y, y_hat)


class VMaxSE(nn.Module):
    def __init__(
        self,
        dim: int = 1,
    ):
        """
        Variance-weighted Maximum Squared Error (VMaxSE) loss function.

        :param dim: The dimension along which to compute the variance.
        """
        super(VMaxSE, self).__init__()
        self._dim: int = dim

    def __call__(self, *args, **kwargs):
        return self.vmaxse(*args, **kwargs)

    def _weight(self, x: Tensor) -> Tensor:
        return torch.repeat_interleave(
            x,
            torch.tensor(range(1, x.size(dim=self._dim) + 1)),
            dim=self._dim,
        )

    def vmaxse(self, y: Tensor, y_hat: Tensor) -> Tensor:
        r = (y - y_hat) ** 2
        w = torch.exp(
            (torch.var(y, dim=self._dim) - torch.var(y_hat, dim=self._dim)) ** 2
        )
        return r.max() * w.max()

    @staticmethod
    def mse(y: Tensor, y_hat: Tensor) -> float:
        return F.mse_loss(y, y_hat).item()

    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:
        return self.vmaxse(y, y_hat)


class MaxAPE(nn.Module):
    def __init__(
        self,
        dim: int = 1,
        benchmark: bool = False,
    ):
        """
        Maximum Absolute Percentage Error (MaxAPE) loss function.

        :param dim: The dimension along which to compute the variance.
        :param benchmark: If True, use the Mean Absolute Percentage Error (MAPE) loss function.
        """
        super(MaxAPE, self).__init__()
        self._dim: int = dim
        self._func: callable = self._mape if benchmark else self._maxape

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def _maxape(self, y: Tensor, y_hat: Tensor) -> Tensor:
        r = (y - y_hat).abs() / y
        w = torch.exp(
            torch.abs(torch.var(y, dim=self._dim) - torch.var(y_hat, dim=self._dim))
        )
        return w.max() * r.max()

    @staticmethod
    def _mape(y: Tensor, y_hat: Tensor) -> Tensor:
        return (y - y_hat).abs() / y

    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:
        return self._func(y, y_hat)

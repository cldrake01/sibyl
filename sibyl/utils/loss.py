import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MaxSE(nn.Module):
    def __init__(
        self,
        dim: int = 1,
    ):
        """
        Variance-weighted Maximum Squared Error (VMaxSE) loss function.

        :param dim: The dimension along which to compute the variance.
        """
        super(MaxSE, self).__init__()
        self._dim: int = dim

    def __call__(self, *args, **kwargs):
        return self.maxse(*args, **kwargs)

    def maxse(self, y_hat: Tensor, y: Tensor) -> Tensor:
        return ((y - y_hat) ** 2).max()

    @staticmethod
    def mse(y_hat: Tensor, y: Tensor) -> float:
        return F.mse_loss(y_hat, y).item()

    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:
        return self.maxse(y_hat, y)


class MaxAE(nn.Module):
    def __init__(
        self,
        dim: int = 1,
    ):
        """
        Variance-weighted Maximum Absolute Error (VMaxAE) loss function.

        :param dim: The dimension along which to compute the variance.
        """
        super(MaxAE, self).__init__()
        self._dim: int = dim

    def __call__(self, *args, **kwargs):
        return self.maxae(*args, **kwargs)

    def _weight(self, x: Tensor) -> Tensor:
        return torch.repeat_interleave(
            x,
            torch.tensor(range(1, x.size(dim=self._dim) + 1)),
            dim=self._dim,
        )

    def maxae(self, y_hat: Tensor, y: Tensor) -> Tensor:
        return (y - y_hat).abs().max()

    @staticmethod
    def mae(y_hat: Tensor, y: Tensor) -> float:
        return F.l1_loss(y_hat, y).item()

    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:
        return self.maxae(y_hat, y)


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

    def vmaxse(self, y_hat: Tensor, y: Tensor) -> Tensor:
        r = (y - y_hat) ** 2
        w = torch.exp(
            (torch.var(y, dim=self._dim) - torch.var(y_hat, dim=self._dim)) ** 2
        )
        return (r * w).max()

    @staticmethod
    def mse(y_hat: Tensor, y: Tensor) -> float:
        return F.mse_loss(y_hat, y).item()

    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:
        return self.vmaxse(y_hat, y)


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

    def vmaxae(self, y_hat: Tensor, y: Tensor) -> Tensor:
        r = (y - y_hat).abs()
        w = torch.exp(
            (torch.var(y, dim=self._dim) - torch.var(y_hat, dim=self._dim)).abs()
        )
        return (r * w).max()

    @staticmethod
    def mae(y_hat: Tensor, y: Tensor) -> float:
        return F.l1_loss(y_hat, y).item()

    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:
        return self.vmaxae(y_hat, y)


class VMaxAPE(nn.Module):
    def __init__(
        self,
        dim: int = 1,
    ):
        """
        Maximum Absolute Percentage Error (MaxAPE) loss function.

        :param dim: The dimension along which to compute the variance.
        """
        super(VMaxAPE, self).__init__()
        self._dim: int = dim

    def __call__(self, *args, **kwargs):
        return self.maxape(*args, **kwargs)

    def maxape(self, y_hat: Tensor, y: Tensor) -> Tensor:
        y += torch.finfo(torch.float32).eps
        r = y - y_hat
        return (r / y).abs().max()

    @staticmethod
    def mape(y_hat: Tensor, y: Tensor) -> float:
        y += torch.finfo(torch.float32).eps
        r = y - y_hat
        return (r / y).abs().mean().item()

    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:
        return self.maxape(y_hat, y)

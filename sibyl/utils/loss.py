import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# def bias_variance_decomposition(y: Tensor, y_hat: Tensor) -> tuple[float, float, float]:
#     """
#     Compute the bias-variance decomposition of a model's predictions.
#
#     :param y: The true target values.
#     :param y_hat: The predicted target values.
#     :return: The bias-variance decomposition.
#     """
#     y, y_hat = y.squeeze(), y_hat.squeeze()
#     y_bar = y.mean()
#     y_hat_bar = y_hat.mean()
#     bias = torch.sum((y_bar - y_hat) ** 2)
#     variance = torch.sum((y_hat - y_hat_bar) ** 2)
#     sum_ = torch.sum(bias + variance).item()
#     return sum_, bias.item(), variance.item()


class Wave(nn.Module):
    def __init__(
        self,
        dim: int = 1,
        weighted: bool = True,
        benchmark: bool = False,
    ):
        super(Wave, self).__init__()
        self._dim = dim

    def __call__(self, *args, **kwargs):
        return self._wave(*args, **kwargs)

    def _wave(self, y: Tensor, y_hat: Tensor) -> Tensor:
        # maybe use a fourier transform to find maximum frequencies for the range
        y = F.interpolate(y, self._dim, mode="linear", align_corners=False)
        y_hat = F.interpolate(y_hat, self._dim, mode="linear", align_corners=False)

        y = y.detach().numpy()
        y_hat = y_hat.detach().numpy()

        # Perform Continuous Wavelet Transform (CWT) for both tensors
        cwt_y, _ = pywt.cwt(y, np.arange(1, 100), "morl")
        cwt_y_hat, _ = pywt.cwt(y_hat, np.arange(1, 100), "morl")

        cwt_y = torch.tensor(cwt_y, requires_grad=True).squeeze()
        cwt_y_hat = torch.tensor(cwt_y_hat, requires_grad=True).squeeze()

        # Calculate the absolute difference between the two CWT results
        cwt_diff = torch.abs(cwt_y - cwt_y_hat)

        cwt_diff = torch.sum(cwt_diff)

        # # Integrate the absolute difference using trapezoidal rule
        # integral = torch.trapezoid(cwt_diff, dim=self._dim)
        #
        # integral = torch.trapezoid(integral)
        #
        # integral = torch.sum(integral)

        # Convert the result to a PyTorch tensor and return
        return cwt_diff

    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:
        return self._wave(y, y_hat)


class VMaxAE(nn.Module):
    def __init__(
        self,
        dim: int = 1,
        benchmark: bool = False,
    ):
        """
        Variance-weighted Maximum Absolute Error (VMaxAE) loss function.

        :param dim: The dimension along which to compute the variance.
        :param benchmark: If True, use the Mean Absolute Error (MAE) loss function.
        """
        super(VMaxAE, self).__init__()
        self._dim: int = dim
        self._func: callable = self._mae if benchmark else self._vmaxae

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)

    def _vmaxae(self, y: Tensor, y_hat: Tensor) -> Tensor:
        r = (y - y_hat).abs()
        w = torch.abs(torch.var(y, dim=self._dim) - torch.var(y_hat, dim=self._dim)) + 1
        # r_max = torch.max(r, dim=self._dim).values
        return r.max() * w.max()

    @staticmethod
    def _mae(y: Tensor, y_hat: Tensor) -> Tensor:
        return F.l1_loss(y, y_hat)

    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:
        return self._func(y, y_hat)


class VMaxSE(nn.Module):
    def __init__(
        self,
        dim: int = 1,
        benchmark: bool = False,
    ):
        """
        Variance-weighted Maximum Squared Error (VMaxSE) loss function.

        :param dim: The dimension along which to compute the variance.
        :param benchmark: If True, use the Mean Squared Error (MSE) loss function.
        """
        super(VMaxSE, self).__init__()
        self._dim: int = dim
        self._func: callable = self._mse if benchmark else self._vmaxse

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)

    def _vmaxse(self, y: Tensor, y_hat: Tensor) -> Tensor:
        r = (y - y_hat) ** 2
        w = torch.exp(
            (torch.var(y, dim=self._dim) - torch.var(y_hat, dim=self._dim)) ** 2
        )
        return w.max() * r.max()

    @staticmethod
    def _mse(y: Tensor, y_hat: Tensor) -> Tensor:
        return F.mse_loss(y, y_hat)

    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:
        return self._func(y, y_hat)


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

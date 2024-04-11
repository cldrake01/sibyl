import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import coherence
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


@torch.jit.script
def bias_variance_decomposition(y: Tensor, y_hat: Tensor) -> tuple[float, float, float]:
    y, y_hat = y.squeeze(), y_hat.squeeze()

    all_pred = y_hat[
        torch.randint_like(y_hat, high=y.size(0), dtype=torch.int),
        torch.randint_like(y_hat, high=y.size(1), dtype=torch.int),
    ]

    avg_expected_loss = torch.mean((all_pred - y) ** 2).item()

    main_predictions = torch.mean(all_pred, dim=0)

    avg_bias = torch.mean((main_predictions - y) ** 2).item()
    avg_var = torch.mean((main_predictions - all_pred) ** 2).item()

    return avg_expected_loss, avg_bias, avg_var


class Wave(nn.Module):
    def __init__(
        self,
        dim: int = 1,
        weighted: bool = True,
        benchmark: bool = False,
    ):
        super(Wave, self).__init__()
        self.dim = dim

    def __call__(self, *args, **kwargs):
        return self._wave(*args, **kwargs)

    def _wave(self, y: Tensor, y_hat: Tensor) -> Tensor:
        # maybe use a fourier transform to find maximum frequencies for the range
        y = F.interpolate(y, self.dim, mode="linear", align_corners=False)
        y_hat = F.interpolate(y_hat, self.dim, mode="linear", align_corners=False)

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
        # integral = torch.trapezoid(cwt_diff, dim=self.dim)
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
        super(VMaxAE, self).__init__()
        self.dim: int = dim
        self.loss: callable = self._mae if benchmark else self._vmaxae

    def __call__(self, *args, **kwargs):
        return self._vmaxae(*args, **kwargs)

    def _vmaxae(self, y: Tensor, y_hat: Tensor) -> Tensor:
        r = (y - y_hat).abs()
        w = torch.abs(torch.var(y, dim=self.dim) - torch.var(y_hat, dim=self.dim)) + 1
        # r_max = torch.max(r, dim=self.dim).values
        return r.max() * w.max()

    @staticmethod
    def _mae(y: Tensor, y_hat: Tensor) -> Tensor:
        return F.l1_loss(y, y_hat)

    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:
        return self.loss(y, y_hat)


class VMaxSE(nn.Module):
    def __init__(
        self,
        dim: int = 1,
        benchmark: bool = False,
    ):
        super(VMaxSE, self).__init__()
        self.dim: int = dim
        self.loss: callable = self._mse if benchmark else self._vmaxse

    def __call__(self, *args, **kwargs):
        return self._vmaxse(*args, **kwargs)

    def _vmaxse(self, y: Tensor, y_hat: Tensor) -> Tensor:
        r = (y - y_hat) ** 2
        w = torch.exp(
            (torch.var(y, dim=self.dim) - torch.var(y_hat, dim=self.dim)) ** 2
        )
        return w.max() * r.max()

    @staticmethod
    def _mse(y: Tensor, y_hat: Tensor) -> Tensor:
        return F.mse_loss(y, y_hat)

    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:
        return self.loss(y, y_hat)


class MaxAPE(nn.Module):
    def __init__(
        self,
        dim: int = 1,
        weighted: bool = True,
        benchmark: bool = False,
    ):
        super(MaxAPE, self).__init__()
        self.dim: int = dim
        self.loss: callable = self._mape if benchmark else self._maxape
        self.weights: callable = self._weights if weighted else lambda x: x

    def __call__(self, *args, **kwargs):
        return self._maxape(*args, **kwargs)

    def _weights(self, t: Tensor) -> Tensor:
        n = t.size(self.dim)
        l = torch.linspace(1, n, n).int()
        t = torch.repeat_interleave(t, l, dim=self.dim)
        t = torch.exp(t)
        return t

    def _maxape(self, y: Tensor, y_hat: Tensor) -> Tensor:
        r = (y - y_hat).abs() / y
        w = torch.exp(
            torch.abs(torch.var(y, dim=self.dim) - torch.var(y_hat, dim=self.dim))
        )
        return w.max() * r.max()

    @staticmethod
    def _mape(y: Tensor, y_hat: Tensor) -> Tensor:
        return (y - y_hat).abs() / y

    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:
        return self.loss(y, y_hat)


class CMaxSE(nn.Module):
    def __init__(
        self,
        dim: int = 1,
        weighted: bool = True,
        benchmark: bool = False,
    ):
        super(CMaxSE, self).__init__()
        self.dim: int = dim
        self.loss: callable = self._mse if benchmark else self._cmaxse
        self.weights: callable = self._weights if weighted else lambda x: x

    def __call__(self, *args, **kwargs):
        return self._cmaxse(*args, **kwargs)

    def _weights(self, t: Tensor) -> Tensor:
        n = t.size(self.dim)
        l = torch.linspace(1, n, n).int()
        t = torch.repeat_interleave(t, l, dim=self.dim)
        t = torch.exp(t)
        return t

    def _cmaxse(self, y: Tensor, y_hat: Tensor) -> Tensor:
        r = (y - y_hat) ** 2
        _, c = coherence(
            y.detach(), y_hat.detach(), axis=self.dim, nperseg=y_hat.size(self.dim)
        )
        return r.max() / c.mean()

    @staticmethod
    def _mse(y: Tensor, y_hat: Tensor) -> Tensor:
        return F.mse_loss(y, y_hat)

    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:
        return self.loss(y, y_hat)


class CMaxAE(nn.Module):
    def __init__(
        self,
        dim: int = 1,
        weighted: bool = True,
        benchmark: bool = False,
    ):
        super(CMaxAE, self).__init__()
        self.dim: int = dim
        self.loss: callable = self._mae if benchmark else self._cmaxae
        self.weights: callable = self._weights if weighted else lambda x: x

    def __call__(self, *args, **kwargs):
        return self._cmaxae(*args, **kwargs)

    def _weights(self, t: Tensor) -> Tensor:
        n = t.size(self.dim)
        l = torch.linspace(1, n, n).int()
        t = torch.repeat_interleave(t, l, dim=self.dim)
        t = torch.exp(t)
        return t

    def _cmaxae(self, y: Tensor, y_hat: Tensor) -> Tensor:
        r = (y - y_hat).abs()
        _, c = coherence(
            y.detach(), y_hat.detach(), axis=self.dim, nperseg=y_hat.size(self.dim)
        )
        return r.max() / c.mean()

    @staticmethod
    def _mae(y: Tensor, y_hat: Tensor) -> Tensor:
        return F.l1_loss(y, y_hat)

    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:
        return self.loss(y, y_hat)

from functools import wraps
from typing import Callable, Any

import pandas as pd
import torch
from scipy.stats import wasserstein_distance
from torch import Tensor

from sibyl.utils.errors import SignatureError


def stats(
    *metrics: Callable[[Tensor, Tensor], float],
) -> callable:
    """
    `stats` is supposed to collect metrics on the two tensors returned by the decorated function.
    *applied are the metrics to be collected.
    The *applied functions should take two tensors and compute their own statistics; they should be self-contained.
    """

    def decorator(func: Callable) -> callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any):
            config, *_ = tuple(filter(lambda x: hasattr(x, "log"), args))
            assert config, SignatureError()
            df: pd.DataFrame = config.metrics
            assert isinstance(
                df, pd.DataFrame
            ), "The provided `Config` object must have a `pd.DataFrame` attribute called `metrics`."
            output = tuple(func(*args, **kwargs))
            for metric in metrics:
                for i, (y, y_hat) in enumerate(t for t in output):
                    df.loc[i, metric.__name__] = metric(y_hat, y)
            return output

        return wrapper

    return decorator


def bias(
    y_hat: Tensor,
    y: Tensor,
) -> float:
    """
    Compute the bias between the actual and predicted values.

    :param y: The actual values.
    :param y_hat: The predicted values.
    """
    return (y - y_hat).abs().mean().item()


def variance(
    y_hat: Tensor,
    y: Tensor,
) -> float:
    """
    Compute the variance between the actual and predicted values.

    :param y: The actual values.
    :param y_hat: The predicted values.
    """
    mu_y_hat = y_hat.mean()
    return ((y_hat - mu_y_hat) ** 2).mean().item()


def error(
    y_hat: Tensor,
    y: Tensor,
) -> float:
    """
    Compute the sum squared error between the actual and predicted values.

    :param y: The actual values.
    :param y_hat: The predicted values.
    """
    return (y - y_hat).abs().sum().item()


def euclidean(
    y_hat: Tensor,
    y: Tensor,
) -> float:
    return (y**2 - y_hat**2).sum().sqrt().item()


def emd(
    y: Tensor,
    y_hat: Tensor,
) -> float:
    p = torch.histogram(y, y.size(1))
    q = torch.histogram(y_hat, y_hat.size(1))
    return wasserstein_distance(p, q)

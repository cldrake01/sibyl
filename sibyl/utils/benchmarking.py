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
            assert isinstance(df, pd.DataFrame), ValueError(
                "The provided `Config` object must have a `pd.DataFrame` attribute called `metrics`."
            )
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

    For bias and variance, we've merely reimplemented mlxtend's bias_variance_decomp function.

    ```py
    avg_expected_loss = np.apply_along_axis(
        lambda x: ((x - y_test) ** 2).mean(), axis=1, arr=all_pred
    ).mean()

    main_predictions = np.mean(all_pred, axis=0)

    avg_bias = np.sum((main_predictions - y_test) ** 2) / y_test.size
    ```

    :param y: The actual values.
    :param y_hat: The predicted values.
    """
    # y_hat_bar = torch.ones_like(y_hat) * y_hat.mean()
    # return ((y_hat_bar - y) ** 2).mean().item()
    return (y_hat - y).abs().mean().item()


def variance(
    y_hat: Tensor,
    y: Tensor,
) -> float:
    """
    Compute the variance between the actual and predicted values.

    For bias and variance, we've merely reimplemented mlxtend's bias_variance_decomp function.

    ```py
    avg_expected_loss = np.apply_along_axis(
        lambda x: ((x - y_test) ** 2).mean(), axis=1, arr=all_pred
    ).mean()

    main_predictions = np.mean(all_pred, axis=0)

    avg_var = np.sum((main_predictions - all_pred) ** 2) / all_pred.size
    ```

    :param y: The actual values.
    :param y_hat: The predicted values.
    """
    # y_hat_bar = torch.ones_like(y_hat) * y_hat.mean()
    # return ((y_hat_bar - y_hat) ** 2).mean().item()

    # Torch's built-in variance function produces identical results to the above.
    return y_hat.var().item()


def std(
    y_hat: Tensor,
    y: Tensor,
) -> float:
    """
    Compute the standard deviation between the actual and predicted values.

    :param y: The actual values.
    :param y_hat: The predicted values.
    """
    return y_hat.std().item()


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

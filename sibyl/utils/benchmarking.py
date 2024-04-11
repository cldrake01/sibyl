from functools import wraps
from typing import Callable, Any, Iterable

from torch import Tensor

from sibyl.utils.config import Config


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
            config: Config = args[-1]
            assert isinstance(config, Config)
            output = tuple(func(*args, **kwargs))
            for metric in metrics:
                config.log.info(f"Computing {metric.__name__}...")
                config.metrics[metric.__name__] = list(
                    metric(y, y_hat) for y, y_hat in (t for t in output)
                )
            return output

        return wrapper

    return decorator


def bias(y: Tensor, y_hat: Tensor) -> float:
    """
    Compute the bias between the actual and predicted values.

    :param y: The actual values.
    :param y_hat: The predicted values.
    """
    return (y.mean() - y_hat.mean()).item()


def variance(y: Tensor, y_hat: Tensor) -> float:
    """
    Compute the variance between the actual and predicted values.

    :param y: The actual values.
    :param y_hat: The predicted values.
    """
    return (y - y_hat).var().item()

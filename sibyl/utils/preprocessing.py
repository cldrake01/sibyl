from typing import Callable, Dict

import numpy as np
import talib
import torch
from torch import Tensor


def indicators(
    time_series: list,
    config: "Config",
) -> tuple[Tensor, Tensor]:
    """
    Calculates technical indicators for a given stock and returns feature and target windows.

    :param time_series: (list): A list of bars for a given stock.
    :param config: (TrainingConfig): A configuration object for time series data.
    :return: tuple[Tensor, Tensor]: Two tensors representing feature windows and target windows.
    """
    # Extracting data points for indicators
    # opens = np.array([bar.open for bar in time_series], dtype=np.float64)
    closes = np.array([bar.close for bar in time_series], dtype=np.float64)
    highs = np.array([bar.high for bar in time_series], dtype=np.float64)
    lows = np.array([bar.low for bar in time_series], dtype=np.float64)
    volumes = np.array([bar.volume for bar in time_series], dtype=np.float64)

    # Defining indicator functions
    indicator_functions = {
        "SMA": lambda x=closes, interval=5: talib.SMA(x, timeperiod=interval),
        "EMA": lambda x=closes, interval=12: talib.EMA(x, timeperiod=interval),
        "WMA": lambda x=closes, interval=12: talib.WMA(x, timeperiod=interval),
        "CCI": lambda interval=20: talib.CCI(highs, lows, closes, timeperiod=interval),
        "ROC": lambda x=closes, interval=12: talib.ROC(x, timeperiod=interval),
        "RSI": lambda x=closes, interval=14: talib.RSI(x, timeperiod=interval),
        "MFI": lambda interval=14: talib.MFI(
            highs, lows, closes, volumes, timeperiod=interval
        ),
        "SAR": lambda a=0.02, m=0.2, interval=0: talib.SAR(
            highs, lows, acceleration=a, maximum=m
        ),
        "ADX": lambda interval=14: talib.ADX(highs, lows, closes, timeperiod=interval),
    }

    if config.included_indicators:
        indicator_functions = {
            func: indicator_functions[func] for func in config.included_indicators
        }

    # Calculating indicators and converting to PyTorch tensors
    XUY = [Tensor(func()).to(config.device) for func in indicator_functions.values()]

    # Stacking indicator tensors and handling NaN values
    XUY = torch.stack(XUY, dim=0)
    XUY = torch.nan_to_num(XUY)

    # Defining window sizes for features and targets
    window_size = config.X_window_size + config.Y_window_size

    # Adding a batch dimension and creating sliding windows
    XUY = XUY.unsqueeze(0)
    XUY = XUY.unfold(-1, window_size, 1)

    # Splitting into features and targets
    X = XUY[..., : config.X_window_size]
    Y = XUY[..., config.X_window_size :]

    return X.squeeze(0), Y.squeeze(0)


def window_function(
    stock_data: dict | list,
) -> Callable[[dict | list, "Config"], tuple[list, list]]:
    """
    A higher-order function that applies the `indicators` function to each stock in the input dictionary and
    aggregates the resulting feature and target windows into lists.

    :param stock_data: (dict or list): A dictionary of stock data or a list of dictionaries of stock data.
    :return: (Callable): A function that applies the `indicators` function to each stock in the input dictionary and
    aggregates the resulting feature and target windows into lists.
    """

    def windows(
        unfiltered: dict,
        config: "Config",
    ) -> tuple[list, list]:
        """
        Applies the `indicators` function to each stock in the input dictionary and aggregates the resulting feature and
        target windows into lists.

        :param unfiltered: (dict): A dictionary of stock data.
        :param config: (TrainingConfig): A configuration object for time series data.
        :return tuple[list, list]: A tuple of lists representing feature windows and target windows.
        """
        Y, X = [], []

        # Filter based upon minimum length of time series
        unfiltered = {
            ticker: time_series
            for ticker, time_series in unfiltered.items()
            if len(time_series) > config.X_window_size + config.Y_window_size
        }

        for ticker, time_series in unfiltered.items():
            x, y = indicators(
                time_series=time_series,
                config=config,
            )
            X.append(x)
            Y.append(y)

        return X, Y

    # Processing each stock in the dictionary
    if isinstance(stock_data, dict):
        return windows
    elif isinstance(stock_data, list):

        def _windows(s_d: list, c: "Config"):
            # Handling case where stock_data is a list of dictionaries
            aggregated = {
                ticker: time_series
                for stock_dict in s_d
                for ticker, time_series in stock_dict.items()
            }
            return windows(aggregated, c)

        return _windows
    raise ValueError(
        f"Expected stock_data to be a dictionary or a list of dictionaries, but got {type(stock_data)}."
    )


def indicator_tensors(
    stock_data: dict | list,
    config: "Config",
) -> tuple[Tensor, Tensor]:
    """
    Applies the `indicators` function to each stock in the input dictionary and aggregates the resulting feature and
    target windows into tensors.

    :param stock_data: (dict or list): A dictionary of stock data or a list of dictionaries of stock data.
    :param config: (TrainingConfig): A configuration object for time series data.
    :return: tuple[Tensor, Tensor]: Two tensors representing feature windows and target windows.
    """
    # Aggregating feature and target windows
    feature_windows_list, target_windows_list = window_function(stock_data)(
        stock_data, config
    )

    # Concatenating all feature and target windows across stocks along the time dimension
    X = torch.cat(feature_windows_list, dim=1).to(config.device)
    Y = torch.cat(target_windows_list, dim=1).to(config.device)

    X = X.permute(1, 2, 0)
    Y = Y.permute(1, 2, 0)

    config.log.info(f"The alpaca dataset has {X.size(0):,} samples.")

    return X, Y


def normalize(
    *tensors: Tensor,
) -> tuple[Tensor, ...]:
    r"""
    See \frac{\left|T\right|}{T}\log_{10}\left(\left|T\right|+1\right), where T is the tensor
    to be normalized.

    This normalization preserves the sign of the tensor whilst normalizing it, as opposed to
    methods available online, wherein the minimum is added to the tensor before normalizing it;
    thereby shifting the tensor to the positive side of the number line, subsequently losing the
    sign of the tensor.

    :param tensors: The tensors to normalize.
    :return: The normalized feature and target tensors.
    """
    return tuple(
        torch.nan_to_num(
            torch.sign(tensor) * torch.log10(torch.abs(tensor) + 1.0).float(), 0.0
        )
        for tensor in tensors
    )

from datetime import datetime
from typing import Callable

import numpy as np
import talib
import torch
from torch import Tensor

from sibyl import TimeSeriesConfig


def indicators(
    time_series: list,
    stock_id: int,
    config: TimeSeriesConfig,
) -> tuple[Tensor, Tensor]:
    """
    Calculates technical indicators for a given stock and returns feature and target windows.

    :param time_series: (list): A list of bars for a given stock.
    :param stock_id: (int): A unique identifier for the stock. Which must be passed to indi
    :param config: (TimeSeriesConfig): A configuration object for time series data.
    :return: tuple[Tensor, Tensor]: Two tensors representing feature windows and target windows.
    """
    # Extracting data points for indicators
    datetimes = [bar.timestamp for bar in time_series]
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
    indicator_tensor_list = [
        Tensor(func()).to(config.device) for func in indicator_functions.values()
    ]

    # Stacking indicator tensors and handling NaN values
    indicator_time_series = torch.stack(indicator_tensor_list, dim=0)
    indicator_time_series = torch.nan_to_num(indicator_time_series)

    # Adding stock ID if required
    if config.include_hashes:
        id_tensor = torch.full(
            (1, indicator_time_series.shape[1]), stock_id, dtype=torch.float64
        )
        indicator_time_series = torch.cat((indicator_time_series, id_tensor), dim=0)

    # Defining window sizes for features and targets
    feature_window_size, target_window_size = (
        config.feature_window_size,
        config.target_window_size,
    )
    total_window_size = feature_window_size + target_window_size

    # Adding a batch dimension and creating sliding windows
    indicator_time_series = indicator_time_series.unsqueeze(0)
    windows = indicator_time_series.unfold(-1, total_window_size, 1)

    # Splitting into features and targets
    feature_windows = windows[..., :feature_window_size]
    target_windows = windows[..., feature_window_size:]

    def temporal_embedding(dt: datetime) -> Tensor:
        """
        Returns a tensor representing the temporal embedding for a given datetime object.

        Note: All values are normalized to be between 0 and 1.
        """
        return Tensor(
            [
                dt.month / 12,
                dt.weekday() / 7,
                dt.day / 31,
                dt.hour / 24,
                dt.minute / 60,
                dt.second / 60,
            ],
        ).to(config.device)

    # Adding temporal embeddings if required
    if config.include_temporal:
        # Creating temporal embeddings
        temporal_embeddings = torch.stack(
            [temporal_embedding(dt) for dt in datetimes],
            dim=0,
        )

    return feature_windows.squeeze(0), target_windows.squeeze(0)


def windows(stock_data: dict, config: TimeSeriesConfig) -> tuple[list, list]:
    """
    Applies the `indicators` function to each stock in the input dictionary and aggregates the resulting feature and
    target windows into lists.

    :param stock_data: (dict): A dictionary of stock data.
    :param config: (TimeSeriesConfig): A configuration object for time series data.
    :return tuple[list, list]: A tuple of lists representing feature windows and target windows.
    """
    target_windows_list, feature_windows_list = [], []

    for ticker, time_series in stock_data.items():
        feature_windows, target_windows = indicators(
            time_series=time_series,
            stock_id=hash(ticker),
            config=config,
        )
        feature_windows_list.append(feature_windows)
        target_windows_list.append(target_windows)

    return feature_windows_list, target_windows_list


def window_function(
    stock_data: dict | list,
) -> Callable[[dict | list, TimeSeriesConfig], tuple[list, list]]:
    """
    A higher-order function that applies the `indicators` function to each stock in the input dictionary and
    aggregates the resulting feature and target windows into lists.

    :param stock_data: (dict or list): A dictionary of stock data or a list of dictionaries of stock data.
    :return: (Callable): A function that applies the `indicators` function to each stock in the input dictionary and
    aggregates the resulting feature and target windows into lists.
    """
    # Processing each stock in the dictionary
    if isinstance(stock_data, dict):
        return windows
    elif isinstance(stock_data, list):

        def list_indicators(s_d: list, c: TimeSeriesConfig):
            # Handling case where stock_data is a list of dictionaries
            aggregated_data = {
                ticker: data
                for stock_dict in s_d
                for ticker, data in stock_dict.items()
            }
            return windows(aggregated_data, c)

        return list_indicators
    else:
        raise ValueError(
            f"Expected stock_data to be a dictionary or a list of dictionaries, but got {type(stock_data)}."
        )


def indicator_tensors(
    stock_data: dict | list,
    config: TimeSeriesConfig,
) -> tuple[Tensor, Tensor]:
    """
    Applies the `indicators` function to each stock in the input dictionary and aggregates the resulting feature and
    target windows into tensors.

    :param stock_data: (dict or list): A dictionary of stock data or a list of dictionaries of stock data.
    :param config: (TimeSeriesConfig): A configuration object for time series data.
    :return: tuple[Tensor, Tensor]: Two tensors representing feature windows and target windows.
    """
    # Aggregating feature and target windows
    feature_windows_list, target_windows_list = window_function(stock_data)(
        stock_data, config
    )

    # Concatenating all feature and target windows across stocks along the time dimension
    feature_windows_tensor = torch.cat(feature_windows_list, dim=1).to(config.device)
    target_windows_tensor = torch.cat(target_windows_list, dim=1).to(config.device)

    feature_windows_tensor = feature_windows_tensor.permute(1, 2, 0)
    target_windows_tensor = target_windows_tensor.permute(1, 2, 0)

    config.log.info(f"feature windows shape: {feature_windows_tensor.shape}")

    return feature_windows_tensor, target_windows_tensor

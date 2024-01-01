import numpy as np
import talib
import torch

from sibyl import TimeSeriesConfig


def indicators(
    time_series: list,
    stock_id: int,
    config: TimeSeriesConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates technical indicators for a given stock and returns feature and target windows.

    :param time_series: (list): A list of bars for a given stock.
    :param stock_id: (int): A unique identifier for the stock. Which must be passed to indi
    :param config: (TimeSeriesConfig): A configuration object for time series data.
    :return: tuple[torch.Tensor, torch.Tensor]: Two tensors representing feature windows and target windows.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"(indicators) time_series: {time_series[:5]}")

    # Extracting data points for indicators
    timestamps = [bar.timestamp for bar in time_series]
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

    # Calculating indicators and converting to PyTorch tensors
    if not config.included_indicators:
        indicator_tensor_list = [
            torch.tensor(np.array(func()), dtype=torch.float64).to(device)
            for func in indicator_functions.values()
        ]
    else:
        indicator_tensor_list = [
            torch.tensor(np.array(indicator_functions[ind]()), dtype=torch.float64).to(
                device
            )
            for ind in config.included_indicators
        ]

    # Stacking indicator tensors and handling NaN values
    indicator_time_series = torch.stack(indicator_tensor_list, dim=0)
    indicator_time_series = torch.nan_to_num(indicator_time_series)

    # Adding stock ID if required
    if config.include_hashes:
        id_tensor = torch.full(
            (1, indicator_time_series.shape[1]), stock_id, dtype=torch.float64
        )
        print(f"(indicators) id_tensor: {id_tensor}")
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

    # Build a tensor of temporal information without concatenating it to the feature and target tensors
    if config.include_temporal:
        temporal_features = [[dt.minute, dt.hour, dt.weekday] for dt in timestamps]

        print(f"(indicators) temporal_features: {temporal_features}")

    return feature_windows.squeeze(0), target_windows.squeeze(0)


def indicator_lists(stock_data: dict, config: TimeSeriesConfig) -> tuple[list, list]:
    target_windows_list, feature_windows_list = [], []

    hashes = {ticker: hash(ticker) for ticker in stock_data.keys()}

    for ticker, stock in stock_data.items():
        feature_windows, target_windows = indicators(
            time_series=stock,
            stock_id=hashes[ticker],
            config=config,
        )
        feature_windows_list.append(feature_windows)
        target_windows_list.append(target_windows)

    return feature_windows_list, target_windows_list


def aggregate_indicators(
    stock_data: dict or list, config: TimeSeriesConfig
) -> tuple[list, list]:
    """
    Applies the `indicators` function to each stock in the input dictionary and aggregates the resulting feature and
    target windows into lists.

    :param stock_data: (dict or list): A dictionary of stock data or a list of dictionaries of stock data.
    :param config: (TimeSeriesConfig): A configuration object for time series data.
    :return: tuple[list, list]: Two lists representing feature windows and target windows.
    """

    # Processing each stock in the dictionary
    if isinstance(stock_data, dict):
        feature_windows_list, target_windows_list = indicator_lists(stock_data, config)
    else:
        # Handling case where stock_data is a list of dictionaries
        aggregated_data = {
            ticker: data
            for stock_dict in stock_data
            for ticker, data in stock_dict.items()
        }

        feature_windows_list, target_windows_list = indicator_lists(
            aggregated_data, config
        )

    return feature_windows_list, target_windows_list


def indicator_tensors(
    stock_data: dict or list,
    config: TimeSeriesConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the `indicators` function to each stock in the input dictionary and aggregates the resulting feature and
    target windows into tensors.

    :param stock_data: (dict or list): A dictionary of stock data or a list of dictionaries of stock data.
    :param config: (TimeSeriesConfig): A configuration object for time series data.
    :return: tuple[torch.Tensor, torch.Tensor]: Two tensors representing feature windows and target windows.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Aggregating feature and target windows
    feature_windows_list, target_windows_list = aggregate_indicators(stock_data, config)

    # Concatenating all feature and target windows across stocks along the time dimension
    feature_windows_tensor = torch.cat(feature_windows_list, dim=1).to(device)
    target_windows_tensor = torch.cat(target_windows_list, dim=1).to(device)

    feature_windows_tensor = feature_windows_tensor.permute(
        1, 2, 0
    )  # New shape for X: [2816814, 60, 8]
    target_windows_tensor = target_windows_tensor.permute(
        1, 2, 0
    )  # New shape for y: [2816814, 15, 8]

    print(
        f"(stock_tensors) new feature_windows_tensor shape: {feature_windows_tensor.shape}"
    )
    print(
        f"(stock_tensors) new target_windows_tensor shape: {target_windows_tensor.shape}"
    )

    return feature_windows_tensor, target_windows_tensor

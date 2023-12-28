import numpy as np
import talib
import torch


def indicators(
    time_series: list,
    stock_id: int,
    include_hashes: bool = False,
    included_indicators: list[str] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates technical indicators for a given stock and returns feature and target windows.

    :param time_series: (list): A list of bars for a given stock.
    :param stock_id: (int): A unique identifier for the stock.
    :param include_hashes: (bool): If True, includes the stock ID in the resulting tensor.
    :param included_indicators: (list[str]): A list of indicators to include in the resulting tensor.
    :return: tuple[torch.Tensor, torch.Tensor]: Two tensors representing feature windows and target windows.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Extracting data points for indicators
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
    if not included_indicators:
        indicator_tensors = [
            torch.tensor(np.array(func()), dtype=torch.float64).to(device)
            for func in indicator_functions.values()
        ]
    else:
        indicator_tensors = [
            torch.tensor(np.array(indicator_functions[ind]()), dtype=torch.float64).to(
                device
            )
            for ind in included_indicators
        ]

    # Stacking indicator tensors and handling NaN values
    indicator_time_series = torch.stack(indicator_tensors, dim=0)
    indicator_time_series = torch.nan_to_num(indicator_time_series)

    # Adding stock ID if required
    if include_hashes:
        id_tensor = torch.full(
            (1, indicator_time_series.shape[1]), stock_id, dtype=torch.float64
        )
        indicator_time_series = torch.cat((id_tensor, indicator_time_series), dim=0)

    # Defining window sizes for features and targets
    input_window_size = 60
    output_window_size = 15
    total_window_size = input_window_size + output_window_size

    # Adding a batch dimension and creating sliding windows
    indicator_time_series = indicator_time_series.unsqueeze(0)
    windows = indicator_time_series.unfold(-1, total_window_size, 1)

    # Splitting into features and targets
    feature_windows = windows[..., :input_window_size]
    target_windows = windows[..., input_window_size:]

    return feature_windows.squeeze(0), target_windows.squeeze(0)


def aggregate_stock_dictionaries(
    stock_data: dict or list, include_hashes: bool = False
) -> tuple[list, list]:
    """
    Applies the `indicators` function to each stock in the input dictionary and aggregates the resulting feature and
    target windows into lists.

    :param stock_data: (dict or list): A dictionary of stock data or a list of dictionaries of stock data.
    :param include_hashes: (bool): If True, includes the stock ID in the resulting tensor.
    :return: tuple[list, list]: Two lists representing feature windows and target windows.
    """
    target_windows_list, feature_windows_list = [], []

    # Processing each stock in the dictionary
    if isinstance(stock_data, dict):
        hashes = {ticker: hash(ticker) for ticker in stock_data.keys()}

        for ticker, stock in stock_data.items():
            feature_windows, target_windows = indicators(
                stock, hashes[ticker], include_hashes
            )
            feature_windows_list.append(feature_windows)
            target_windows_list.append(target_windows)
    else:
        # Handling case where stock_data is a list of dictionaries
        aggregated_data = {
            ticker: data
            for stock_dict in stock_data
            for ticker, data in stock_dict.items()
        }
        hashes = {ticker: hash(ticker) for ticker in aggregated_data.keys()}

        for ticker, stock in aggregated_data.items():
            feature_windows, target_windows = indicators(stock, hashes[ticker])
            feature_windows_list.append(feature_windows)
            target_windows_list.append(target_windows)

    return feature_windows_list, target_windows_list


def stock_tensors(
    stock_data: dict or list,
    include_hashes: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the `indicators` function to each stock in the input dictionary and aggregates the resulting feature and
    target windows into tensors.

    :param stock_data: (dict or list): A dictionary of stock data or a list of dictionaries of stock data.
    :param include_hashes: (bool): If True, includes the stock ID in the resulting tensor.
    :return: tuple[torch.Tensor, torch.Tensor]: Two tensors representing feature windows and target windows.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Aggregating feature and target windows
    feature_windows_list, target_windows_list = aggregate_stock_dictionaries(
        stock_data, include_hashes
    )

    # Concatenating all feature and target windows across stocks along the time dimension
    feature_windows_tensor = torch.cat(feature_windows_list, dim=1).to(device)
    target_windows_tensor = torch.cat(target_windows_list, dim=1).to(device)

    feature_windows_tensor.permute(1, 2, 0)  # New shape for X: [2816814, 60, 8]
    target_windows_tensor.permute(1, 2, 0)  # New shape for y: [2816814, 15, 8]

    return feature_windows_tensor, target_windows_tensor

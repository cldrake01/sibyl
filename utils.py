import os
from datetime import datetime
from enum import Enum
from pprint import pprint

import torch
import talib
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import seaborn as sns
import torch.optim as optim
import matplotlib.pyplot as plt
from alpaca.data.historical import *
from alpaca.data import StockBarsRequest, TimeFrame
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

API_KEY = "PK3D0P0EF5NVU7LKHY76"
API_SECRET = "X2kmdCqfYzGxaCYG2C3UhQ9DqHT9bYhYUhXM2g6G"


def indicators(time_series: dict, _indicators: list = None) -> np.ndarray:
    """
    Generate technical indicators for a given time series.
    """

    time_series = list(time_series.values())[0]

    closes = np.array([bar.close for bar in time_series], dtype=np.float64)
    opens = np.array([bar.open for bar in time_series], dtype=np.float64)
    highs = np.array([bar.high for bar in time_series], dtype=np.float64)
    lows = np.array([bar.low for bar in time_series], dtype=np.float64)
    volumes = np.array([bar.volume for bar in time_series], dtype=np.float64)

    indicator_functions = {
        "SMA": lambda x=closes, interval=5: talib.SMA(x, timeperiod=interval),
        "EMA": lambda x=closes, interval=12: talib.EMA(x, timeperiod=interval),
        "WMA": lambda x=closes, interval=12: talib.WMA(x, timeperiod=interval),
        "BBANDS": lambda x=closes, interval=20: talib.BBANDS(
            x, timeperiod=interval, nbdevup=1.5, nbdevdn=1.5
        ),
        "CCI": lambda interval=20: talib.CCI(highs, lows, closes, timeperiod=interval),
        "ROC": lambda x=closes, interval=12: talib.ROC(x, timeperiod=interval),
        "RSI": lambda x=closes, interval=14: talib.RSI(x, timeperiod=interval),
        "STOCH": lambda fk=5, sk=3: talib.STOCH(
            highs, lows, closes, fastk_period=fk, slowk_period=sk
        ),
        "MFI": lambda interval=14: talib.MFI(
            highs, lows, closes, volumes, timeperiod=interval
        ),
        "SAR": lambda a=0.02, m=0.2, interval=0: talib.SAR(
            highs, lows, acceleration=a, maximum=m
        ),
    }

    indicator_time_series = np.array([], dtype=np.float64)

    if not _indicators:
        for func in indicator_functions.values():
            result = np.array(func())
            indicator_time_series = np.concatenate(
                (indicator_time_series, result.flatten())
            )
    else:
        for indicator in _indicators:
            result = np.array(indicator_functions[indicator]())
            indicator_time_series = np.concatenate(
                (indicator_time_series, result.flatten())
            )

    return indicator_time_series


def alpaca_time_series(
    stocks: list[str], start: datetime or str, end: datetime or str
) -> dict:
    """
    Retrieve data from the Alpaca API.
    """

    if isinstance(start, datetime):
        start = start.strftime("%Y-%m-%d %H:%M:%S")

    if isinstance(end, datetime):
        end = end.strftime("%Y-%m-%d %H:%M:%S")

    client = StockHistoricalDataClient(
        API_KEY,
        API_SECRET,
        url_override="https://data.alpaca.markets",
    )

    params = StockBarsRequest(
        symbol_or_symbols=stocks,
        start=start,
        end=end,
        timeframe=TimeFrame.Hour,
    )

    return client.get_stock_bars(params).data


if __name__ == "__main__":
    data = alpaca_time_series(["AAPL"], "2021-01-01 00:00:00", "2021-01-31 00:00:00")
    data = indicators(data)
    pprint(list(data))

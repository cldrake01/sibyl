import os
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


def day_indicators(week: dict):
    """
    Calculate day trading indicators with adapted parameters.

    Args:
    df (pd.DataFrame): DataFrame containing OHLCV (Open, High, Low, Close, Volume) data.
    symbol (str): The symbol for which the indicators are calculated.

    Returns:
    pd.DataFrame: Updated DataFrame with day trading indicators.
    """
    close = np.array([day.close for day in week])
    high = np.array([day.high for day in week])
    low = np.array([day.low for day in week])
    volume = np.array([day.volume for day in week])

    ind = list()

    # Moving Averages (SMA)
    ind.append(talib.SMA(close, timeperiod=5))
    ind.append(talib.SMA(close, timeperiod=10))

    # Bollinger Bands (BBANDS)
    upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=1.5, nbdevdn=1.5)
    ind.append(upper)
    ind.append(middle)
    ind.append(lower)

    # Commodity Channel Index (CCI)
    ind.append(talib.CCI(high, low, close, timeperiod=10))

    # Rate of Change (ROC)
    ind.append(talib.ROC(close, timeperiod=5))
    ind.append(talib.ROC(close, timeperiod=10))

    # Relative Strength Index (RSI)
    ind.append(talib.RSI(close, timeperiod=5))
    ind.append(talib.RSI(close, timeperiod=10))

    # Stochastic Oscillator (STOCH)
    fastk, slowk = talib.STOCH(high, low, close, fastk_period=5, slowk_period=3)
    ind.append(fastk)
    ind.append(slowk)

    # Money Flow Index (MFI)
    ind.append(talib.MFI(high, low, close, volume, timeperiod=5))
    ind.append(talib.MFI(high, low, close, volume, timeperiod=10))

    # Parabolic SAR (SAR)
    ind.append(talib.SAR(high, low, acceleration=0.02, maximum=0.2))

    return ind


def data(stocks: list, start: str = None, end: str = None, timeframe: str = None):
    """
    Retrieve data from the Alpaca API.
    """

    if start is None:
        start = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=28)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    if end is None:
        end = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=24)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    print(f"start: {start}, end: {end}, {pd.Timestamp.now(tz='UTC')}")

    if timeframe is None:
        timeframe = TimeFrame.Minute

    api_key = "PK3D0P0EF5NVU7LKHY76"
    api_secret = "X2kmdCqfYzGxaCYG2C3UhQ9DqHT9bYhYUhXM2g6G"

    client = StockHistoricalDataClient(
        api_key,
        api_secret,
        url_override="https://data.alpaca.markets",
    )

    params = StockBarsRequest(
        symbol_or_symbols=stocks,
        start=start,
        end=end,
        timeframe=timeframe,
    )

    return client.get_stock_bars(params)


if __name__ == "__main__":
    print(data(["AAPL"]))

    # # Retrieve 20 days (one month) of price data.
    # start_date = f"2022-01-01 00:00:00"
    # end_date = f"2022-01-21 00:00:00"
    #
    # params = StockBarsRequest(
    #     symbol_or_symbols=["AAPL"],
    #     start=start_date,
    #     end=end_date,
    #     timeframe=TimeFrame.Minute,
    # )
    #
    # api_key = "PK3D0P0EF5NVU7LKHY76"
    # api_secret = "X2kmdCqfYzGxaCYG2C3UhQ9DqHT9bYhYUhXM2g6G"
    #
    # client = StockHistoricalDataClient(
    #     api_key,
    #     api_secret,
    #     url_override="https://data.alpaca.markets",
    # )
    #
    # print(client.get_stock_bars(params))

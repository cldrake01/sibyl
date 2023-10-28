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

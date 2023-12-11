import os
import talib
import torch
import pickle
import numpy as np
import pandas as pd
import torch.nn as nn
import seaborn as sns
import torch.optim as optim
import matplotlib.pyplot as plt

from torch import nn
from tqdm import tqdm
from enum import Enum
from pprint import pprint
from itertools import chain
from dataclasses import dataclass
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from alpaca.data import StockBarsRequest, TimeFrame
from alpaca.data.historical import StockHistoricalDataClient

from log import *
from sp import sp_tickers

API_KEY = "PK3D0P0EF5NVU7LKHY76"
API_SECRET = "X2kmdCqfYzGxaCYG2C3UhQ9DqHT9bYhYUhXM2g6G"


class MetaLabeler(nn.Module):
    """
    A transformer model for determining the magnitude of Vec's and RVec's proposed trades. MetaLabeler is trained
    upon Vec's and RVec's predictions.
    """

    def __init__(self, *args, **kwargs):
        super(MetaLabeler, self).__init__()


@dataclass
class Bar:
    """
    A bar is a single unit of time series data.
    """

    open: float
    high: float
    low: float
    close: float
    volume: int

    def __dict__(self):
        return {
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


class NullLogger:
    def __init__(self, *args, **kwargs):
        ...

    def info(self, *args, **kwargs):
        ...

    def debug(self, *args, **kwargs):
        ...

    def warning(self, *args, **kwargs):
        ...

    def error(self, *args, **kwargs):
        ...


class Informer(nn.Module):
    def __init__(
        self, input_size, output_size, d_model, n_heads, e_layers, d_layers, dropout=0.1
    ):
        super(Informer, self).__init__()
        # Encoder and Decoder sizes
        self.encoder_input_size = input_size
        self.decoder_input_size = output_size

        # Encoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True
            ),
            num_layers=e_layers,
        )

        # Decoder
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True
            ),
            num_layers=d_layers,
        )

        # Linear layers for adjusting dimensions
        self.encoder_embedding = nn.Linear(self.encoder_input_size, d_model)
        self.decoder_embedding = nn.Linear(self.decoder_input_size, d_model)
        self.output_layer = nn.Linear(d_model, self.decoder_input_size)

    def forward(self, src, tgt):
        # Embedding and encoding the source sequence
        src = self.encoder_embedding(src)
        memory = self.encoder(src)

        # Embedding and decoding the target sequence
        tgt = self.decoder_embedding(tgt)
        output = self.decoder(tgt, memory)

        # Final linear layer
        output = self.output_layer(output)
        return output


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
    )

    # url_override="https://data.alpaca.markets",

    params = StockBarsRequest(
        symbol_or_symbols=stocks,
        start=start,
        end=end,
        timeframe=TimeFrame.Minute,
    )

    return client.get_stock_bars(params).data


def fetch_data(
    years: float,
    max_workers: int = len(sp_tickers) // 2,
    log: logging.Logger = NullLogger(),
) -> list:
    # Retrieve data from the Alpaca API
    log.info("Retrieving data from the Alpaca API...")
    log.info(f"Maximum Worker-threads: {max_workers}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                alpaca_time_series,
                [ticker],
                datetime.today() - timedelta(365 * years),
                datetime.today(),
            )
            for ticker in sp_tickers
        ]

    # Collecting results
    data = [future.result() for future in futures]
    log.info("Retrieved data from the Alpaca API.")

    return data


def window_tensors(
    tensor, input_window_size, output_window_size
) -> tuple[torch.Tensor, torch.Tensor]:
    total_window_size = input_window_size + output_window_size

    # Add a batch dimension if not present
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)

    # Use unfold to create sliding windows
    windows_ = tensor.unfold(-1, total_window_size, 1)

    # Split into features and targets
    feature_windows = windows_[..., :input_window_size]
    target_windows = windows_[..., input_window_size:]

    return feature_windows, target_windows


def indicators(
    time_series: list,
    stock_id: int,
    include_hashes: bool = False,
    included_indicators: list[str] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate technical indicators for a given time series and create feature and target windows.

    Args:
        time_series (list): A list of time series data points, typically as 'Bar' objects.
        stock_id (int): Unique identifier for the stock.
        include_hashes (bool): If True, includes the stock ID in the resulting tensor.
        included_indicators (list[str], optional): List of indicators to be included. If None, all indicators are used.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Two tensors representing feature windows and target windows.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Extracting data points for indicators
    closes = np.array([bar.close for bar in time_series], dtype=np.float64)
    opens = np.array([bar.open for bar in time_series], dtype=np.float64)
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
    }

    # Calculating indicators and converting to PyTorch tensors
    indicator_tensors = [
        torch.tensor(np.array(func()), dtype=torch.float64).to(device)
        for func in indicator_functions.values()
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


def stock_tensors(
    stock_data: dict or list,
    include_hashes: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the `indicators` function to each stock in the input dictionary and aggregates the resulting feature and
    target windows into tensors.

    Args:
        stock_data (dict): Dictionary where keys are stock names and values are time series data for each stock.
        include_hashes (bool): Whether to include the hash of each stock's name in the resulting tensor.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Two tensors containing all feature windows and target windows for all stocks.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_windows_list = []
    target_windows_list = []

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

    # Concatenating all feature and target windows across stocks along the time dimension
    feature_windows_tensor = torch.cat(feature_windows_list, dim=1).to(device)
    target_windows_tensor = torch.cat(target_windows_list, dim=1).to(device)

    feature_windows_tensor = feature_windows_tensor.transpose(0, 1).transpose(
        1, 2
    )  # New shape for X: [2816814, 60, 8]
    target_windows_tensor = target_windows_tensor.transpose(0, 1).transpose(
        1, 2
    )  # New shape for y: [2816814, 15, 8]

    return feature_windows_tensor, target_windows_tensor


def train(
    X: torch.Tensor,  # Feature windows tensor
    y: torch.Tensor,  # Target windows tensor
    model,  # Your TimeSeriesTransformer model
    normalized: bool = True,
    epochs: int = 1_000_000,
    log: logging.Logger = NullLogger(),
    batch_size: int = 1,
    learning_rate: float = 0.001,
) -> None:
    """
    Train the Time Series Transformer model.

    Args:
        X (torch.Tensor): The input feature tensor.
        y (torch.Tensor): The target tensor.
        model (TimeSeriesTransformerForPrediction): The time series transformer model.
        normalized (bool): Whether the data is normalized.
        epochs (int): Number of training epochs.
        log (logging.Logger): Logger for training progress.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for the optimizer.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info(f"X.shape: {X.shape}, y.shape: {y.shape}")

    # Generate dummy past_time_features and past_observed_mask
    # num_samples, num_time_steps, num_features = X.shape
    #
    # time_steps_tensor = torch.arange(1, num_time_steps + 1).view(1, num_time_steps, 1)
    # past_time_features = (
    #     time_steps_tensor.expand(-1, -1, num_features).float().to(device)
    # )
    # past_observed_mask = torch.ones(num_time_steps, num_features).float().to(device)

    # log.info(f"past_time_features.shape: {past_time_features.size()}")
    # log.info(f"past_observed_mask.shape: {past_observed_mask.size()}")

    # Normalizing data if required
    if normalized:
        X = torch.nan_to_num(torch.log10(X.detach().clone() + 1.0), nan=0.0).float()
        y = torch.nan_to_num(torch.log10(y.detach().clone() + 1.0), nan=0.0).float()

    # Creating a TensorDataset and DataLoader
    log.info("Creating a TensorDataset and DataLoader...")
    dataset = TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    log.info("TensorDataset and DataLoader created.")

    # Model, loss function, and optimizer
    log.info("Moving model to device...")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    log.info("Model moved to device.")

    # Training loop
    best_loss = float("inf")
    early_stopping_patience = 10
    patience_counter = 0

    epoch_train_losses, epoch_val_losses = [], []

    for epoch in tqdm(range(epochs), desc="Training"):
        model.train()
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.float().to(device), y_batch.float().to(device)
            optimizer.zero_grad()

            # log.info(f"X_batch.shape: {X_batch.size()}")
            # log.info(f"y_batch.shape: {y_batch.size()}")

            output = model(
                X_batch, y_batch[:, :-1, :]
            )  # Exclude last time step from target for input
            y_true = y_batch[
                :, 1:, :
            ]  # Exclude first time step from target for loss calculation

            loss = criterion(output, y_true)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation step
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                val_output = model(X_val, y_val[:, :-1, :])
                y_true_val = y_val[:, 1:, :]

                val_loss = criterion(val_output, y_true_val)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        # Early stopping check
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                log.info("Early stopping triggered.")
                break

        # Record the average loss of this epoch
        epoch_train_losses.append(avg_train_loss)
        epoch_val_losses.append(avg_val_loss)

        log.info(
            f"Epoch {epoch}: Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}"
        )

    # Save the model
    if not os.path.exists("models"):
        os.mkdir("models")
    torch.save(model.state_dict(), "models/transformer.pt")
    log.info("Model saved.")

    # Plot loss
    if not os.path.exists("plots"):
        os.mkdir("plots")
    plt.plot(epoch_train_losses, label="Train Loss")
    plt.plot(epoch_val_losses, label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig("plots/loss.png")
    plt.clf()

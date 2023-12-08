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
from sklearn.model_selection import train_test_split

from torch import nn
from tqdm import tqdm
from enum import Enum
from pprint import pprint
from itertools import chain
from dataclasses import dataclass
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error
from concurrent.futures import ThreadPoolExecutor
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
from classes import Vec, TimeSeriesDataset, NullLogger, Bar

API_KEY = "PK3D0P0EF5NVU7LKHY76"
API_SECRET = "X2kmdCqfYzGxaCYG2C3UhQ9DqHT9bYhYUhXM2g6G"


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


def stock_tensors(stock_data) -> torch.Tensor:
    """
    Applies the `indicators` function to each stock in the input dictionary, pads each resulting tensor to the
    maximum dimension among them, and then stacks them into a single tensor.

    :param stock_data: A dictionary where the keys are stock names and the values are time series data for each stock.

    :returns torch.Tensor: A tensor that is the result of stacking the padded tensors obtained from each stock's time
    series data.
    """
    indicator_tensors_list = []

    if type(stock_data) is dict:
        # Generate unique integer IDs for each stock id
        # Note: Including a Hash makes the final tensor very difficult to read and interpret, as the values are
        # represented in scientific notation. E.g., 2.3759e+01 (2.3759*10^1) instead of 23.759
        hashes = {ticker: i for i, ticker in enumerate(stock_data.keys())}

        indicator_tensors_list = [
            indicators(stock, hashes[ticker]) for ticker, stock in stock_data.items()
        ]
    else:
        # Aggregate all dictionaries contained within the list.
        aggregated_data = {}
        for stock_dict in stock_data:
            for ticker, data in stock_dict.items():
                aggregated_data[ticker] = data

        # New: Generate unique integer IDs for each stock id in the aggregated data
        hashes = {ticker: i for i, ticker in enumerate(aggregated_data.keys())}

        indicator_tensors_list = [
            indicators(stock, hashes[ticker])
            for ticker, stock in aggregated_data.items()
        ]

    # Padding and stacking process remains the same as before
    max_dim = max([stock.shape[1] for stock in indicator_tensors_list])
    padded_indicator_tensors = [
        torch.nn.functional.pad(
            stock, (max_dim - stock.shape[1], 0), mode="constant", value=0.0
        )
        for stock in indicator_tensors_list
    ]
    return torch.stack(padded_indicator_tensors, dim=0)


def indicators(
    time_series: list, stock_id: int, _indicators: list = None
) -> torch.Tensor:
    """
    Generate technical indicators for a given time series.
    """

    closes = np.array([bar.close for bar in time_series], dtype=np.float64)
    opens = np.array([bar.open for bar in time_series], dtype=np.float64)
    highs = np.array([bar.high for bar in time_series], dtype=np.float64)
    lows = np.array([bar.low for bar in time_series], dtype=np.float64)
    volumes = np.array([bar.volume for bar in time_series], dtype=np.float64)

    indicator_functions = {
        "SMA": lambda x=closes, interval=5: talib.SMA(x, timeperiod=interval),
        "EMA": lambda x=closes, interval=12: talib.EMA(x, timeperiod=interval),
        "WMA": lambda x=closes, interval=12: talib.WMA(x, timeperiod=interval),
        # "BBANDS" has shape (3, t), as opposed to (t)
        # "BBANDS": lambda x=closes, interval=20: talib.BBANDS(
        #     x, timeperiod=interval, nbdevup=1.5, nbdevdn=1.5
        # ),
        "CCI": lambda interval=20: talib.CCI(highs, lows, closes, timeperiod=interval),
        "ROC": lambda x=closes, interval=12: talib.ROC(x, timeperiod=interval),
        "RSI": lambda x=closes, interval=14: talib.RSI(x, timeperiod=interval),
        # "STOCH" has shape (2, t), as opposed to (t)
        # "STOCH": lambda fk=5, sk=3: talib.STOCH(
        #     highs, lows, closes, fastk_period=fk, slowk_period=sk
        # ),
        "MFI": lambda interval=14: talib.MFI(
            highs, lows, closes, volumes, timeperiod=interval
        ),
        "SAR": lambda a=0.02, m=0.2, interval=0: talib.SAR(
            highs, lows, acceleration=a, maximum=m
        ),
    }

    indicator_tensors = []

    if not _indicators:
        for func in indicator_functions.values():
            result = np.array(func())
            indicator_tensor = torch.tensor(result, dtype=torch.float64)
            indicator_tensors.append(indicator_tensor)
    else:
        for indicator in _indicators:
            result = np.array(indicator_functions[indicator]())
            indicator_tensor = torch.tensor(result, dtype=torch.float64)
            indicator_tensors.append(indicator_tensor)

    indicator_time_series = torch.stack(indicator_tensors, dim=0)

    indicator_time_series = torch.nan_to_num(
        indicator_time_series,
        nan=0.0,
        posinf=indicator_time_series.max(),
        neginf=indicator_time_series.min(),
    )

    # New: Create a tensor for the stock ID and prepend it to the indicator time series
    id_tensor = torch.full(
        (1, indicator_time_series.shape[1]), stock_id, dtype=torch.float64
    )
    indicator_time_series = torch.cat((id_tensor, indicator_time_series), dim=0)

    return indicator_time_series


def create_windows(
    data,
    input_window_size=60,
    output_window_size=15,
    log: logging.Logger = NullLogger(),
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create input and output windows for the Vec model.

    :param data: A tensor of shape (n, t, d) where n is the number of stocks, t is the number of time steps, and d is the
    number of technical indicators.
    :param input_window_size: The number of time steps to use as input.
    :param output_window_size: The number of time steps to use as output.
    :param log: A logger object.
    :return: A tuple of tensors where the first tensor is the input data and the second tensor is the output data.
    """
    input_data, output_data = [], []
    insufficient_data_count = 0

    # Loop over each stock
    for stock_index in range(data.shape[0]):
        stock_data = data[stock_index]  # Get data for the current stock
        stock_input, stock_output = [], []

        # Check if there are enough time steps for the current stock
        if stock_data.shape[1] >= input_window_size + output_window_size:
            # Create windows for the current stock
            for i in range(
                stock_data.shape[1] - input_window_size - output_window_size + 1
            ):
                input_window = stock_data[:, i : i + input_window_size]
                output_window = stock_data[
                    :,
                    i + input_window_size : i + input_window_size + output_window_size,
                ]
                stock_input.append(input_window)
                stock_output.append(output_window)

            input_data.extend(stock_input)
            output_data.extend(stock_output)
        else:
            insufficient_data_count += 1
            log.error(
                f"Insufficient data for stock index {stock_index} with time steps: {stock_data.shape[1]}."
            )

        log.info(
            f"Stock index {stock_index}: Created {len(stock_input)} input windows and {len(stock_output)} output windows."
        )

    if input_data and output_data:
        return torch.stack(input_data, dim=0), torch.stack(output_data, dim=0)
    else:
        log.error(
            f"No windows were created. Insufficient data stocks count: {insufficient_data_count}."
        )
        return torch.tensor([]), torch.tensor([])


def train(
    time_series: list,
    normalized: bool = True,
    epochs: int = 1_000_000,
    log: logging.Logger = NullLogger(),
    batch_size: int = 32,
) -> None:
    """
    Train the Vec model.
    """

    if torch.cuda.is_available():
        device = torch.device("cuda")
        log.info("CUDA is available.")
    else:
        device = torch.device("cpu")
        log.info("CUDA is not available.")

    log.info("Generating technical indicators for each stock...")
    time_series_tensor = stock_tensors(time_series)
    log.info("Generated technical indicators for each stock.")

    log.info("Normalizing the data...")
    norm_data = torch.nan_to_num(
        torch.log(time_series_tensor.detach().clone() + 1.0), nan=0.0
    )
    log.info("Normalized the data.")

    # Prepare dataset
    dataset = TimeSeriesDataset(norm_data if normalized else time_series_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Prepare data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Instantiate the model
    transformer = Vec(dataset[0][0].shape[-1]).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(transformer.parameters(), lr=0.001)

    graph_loss = []
    best_loss = float("inf")
    early_stopping_patience = 10
    patience_counter = 0

    for epoch in tqdm(range(epochs)):
        transformer.train()
        total_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = transformer(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        transformer.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(
                    device
                )
                val_output = transformer(X_val_batch)
                val_loss = criterion(val_output, y_val_batch)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter > early_stopping_patience:
                log.info(f"Early stopping triggered at epoch {epoch}.")
                break

        graph_loss.append(avg_train_loss)

    if not os.path.exists("plots"):
        os.mkdir("plots")

    plt.plot(graph_loss)
    plt.title("Loss vs. Epochs (Normalized)" if normalized else "Loss vs. Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("plots/loss.png")
    plt.clf()

    log.info("Saving transformer.pt...")
    if not os.path.exists("models"):
        os.mkdir("models")

    torch.save(transformer.state_dict(), "models/transformer.pt")
    log.info("Saved transformer.pt")


if __name__ == "__main__":
    # data = torch.Tensor([[x for x in range(76)] for _ in range(76)])
    # pprint(data)
    # pprint(data[:, 60:])

    log = setup_logging("utils.py")

    # def windows(time_series_tensor: torch.Tensor, input_window=60, output_window=15):
    #     # Flatten the 'stocks' and 'indicators' dimensions
    #     input_shape = (
    #         time_series_tensor.size(0) * time_series_tensor.size(1),
    #         time_series_tensor.size(2),
    #     )
    #     data_flattened = time_series_tensor.view(input_shape)
    #
    #     total_windows = data_flattened.size(1) - input_window - output_window + 1
    #
    #     X = torch.empty((total_windows, input_shape[0], input_window))
    #     y = torch.empty((total_windows, input_shape[0], output_window))
    #
    #     for i in range(total_windows):
    #         X[i] = data_flattened[:, i : i + input_window]
    #         y[i] = data_flattened[
    #             :, i + input_window : i + input_window + output_windowZzzz
    #         ]
    #
    #     return X, y
    #
    time_series = stock_tensors(fetch_data(years=0.1))

    log.info(time_series)
    log.info(time_series.size())
    print([time_series[i][0] for i in range(len(time_series))])

    # X, y = windows(time_series)
    #
    # log.info(X)
    # log.info(X.size())
    # log.info(y)
    # log.info(y.size())

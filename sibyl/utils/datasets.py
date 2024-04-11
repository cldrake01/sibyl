import os
import pickle
from typing import Callable

import pandas as pd
import torch
from torch import Tensor

from sibyl.utils.logging import find_root_dir
from sibyl.utils.preprocessing import indicator_tensors
from sibyl.utils.retrieval import fetch_data


def cache(
    func: Callable[["Config", ...], tuple[Tensor, Tensor]]
) -> Callable[["Config", ...], tuple[Tensor, Tensor]]:
    """
    Cache the results of a dataset function to disk.
    This is particularly useful for dataset functions that make API calls, e.g., Alpaca.

    :param func: The function to cache
    """

    def wrapper(*args, **kwargs):
        config = args[0]
        root = find_root_dir(os.path.dirname(__file__))
        file_path = f"{root}/assets/pkl/{func.__name__}.pkl"
        if os.path.exists(file_path):
            config.log.info(f"Loading cached data from {file_path}...")
            with open(file_path, "rb") as f:
                return pickle.load(f)
        else:
            config.log.info(f"Data not found at {file_path}. Fetching data...")
            data = func(*args, **kwargs)
            config.log.info(f"Caching data to {file_path}...")
            with open(file_path, "wb") as f:
                pickle.dump(data, f)
            return data

    return wrapper


@cache
def alpaca(config: "Config") -> tuple[Tensor, Tensor]:
    time_series = fetch_data(config=config)
    features, targets = indicator_tensors(time_series, config=config)
    return features, targets


@cache
def ett(
    config: "Config",
    directory: str | None = None,
    file: str = "ETTh1.csv",
) -> tuple[Tensor, Tensor]:
    """
    Parse the ETT CSVs and return tensors of shape (batch, features, time).

    :param config: Configuration object
    :param directory: Directory containing the CSVs
    :param file: The CSV file to parse

    :return: Feature windows, target windows
    """
    directory = directory or os.path.join(
        find_root_dir(os.path.dirname(__file__)), "assets", "datasets", "ett"
    )

    # Load the CSVs
    X = pd.read_csv(f"{directory}/{file}")

    # Remove the date column
    X = X.drop(columns=["date"])

    # Convert to tensors
    X = torch.tensor(X.to_numpy()).float()

    # Reshape the tensors
    X = X.permute(1, 0)

    # Window the tensors
    feature_window_size, target_window_size = (
        config.feature_window_size,
        config.target_window_size,
    )
    total_window_size = feature_window_size + target_window_size

    X = X.unfold(-1, total_window_size, 1)

    # (features, batch, time) -> (batch, time, features)
    X = X.permute(1, 2, 0)

    # Split into features and targets
    train = X[:, :feature_window_size, :]
    test = X[:, feature_window_size:, :]

    return train, test


@cache
def eld(
    config: "Config",
    directory: str | None = None,
    file: str = "LD2011_2014.txt",
) -> tuple[Tensor, Tensor]:
    """
    Parse the ELD text file and return tensors of shape (batch, features, time).

    :param config: Configuration object
    :param directory: Directory containing the text file
    :param file: The text file to parse
    """
    directory = directory or os.path.join(
        find_root_dir(os.path.dirname(__file__)), "assets", "datasets", "eld"
    )

    X = pd.read_csv(f"{directory}/{file}", delimiter=";", decimal=",")

    # Remove the date column
    X = X.drop(columns=["Unnamed: 0"])

    # Convert to tensors
    X = torch.tensor(X.to_numpy()).float()

    # Reshape the tensors
    X = X.permute(1, 0)

    # Window the tensors
    feature_window_size, target_window_size = (
        config.feature_window_size,
        config.target_window_size,
    )
    total_window_size = feature_window_size + target_window_size

    X = X.unfold(-1, total_window_size, 1)

    # (features, batch, time) -> (batch, time, features)
    X = X.permute(1, 2, 0)

    # Split into features and targets
    train = X[:, :feature_window_size, :]
    test = X[:, feature_window_size:, :]

    # Due to its sheer size, we'll only use the first 15 features
    train = train[..., :15]
    test = test[..., :15]

    return train, test

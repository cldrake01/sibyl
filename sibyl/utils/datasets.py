import os
import pickle
from functools import wraps
from typing import Callable

import pandas as pd
import torch
from torch import Tensor

from sibyl.utils.logging import find_root_dir
from sibyl.utils.preprocessing import indicator_tensors
from sibyl.utils.retrieval import fetch_data


class SignatureError(Exception):
    def __init__(self):
        self.message = "The provided callable must have a Config object as an argument."
        super().__init__(self.message)


def cache(
    func: Callable[["Config", ...], tuple[Tensor, Tensor]]
) -> Callable[["Config", ...], tuple[Tensor, Tensor]]:
    """
    Cache the results of a dataset function to disk.
    This is particularly useful for dataset functions that make API calls, e.g., Alpaca.

    :param func: The function to cache
    """

    @wraps(func)
    def _cache(*args, **kwargs):
        config, *_ = tuple(filter(lambda x: hasattr(x, "log"), args))
        assert config, SignatureError()
        root = find_root_dir(os.path.dirname(__file__))
        file_path = f"{root}/assets/pkl/{func.__name__}.pkl"
        if os.path.exists(file_path):
            config.log.info(f"Loading cached data from {file_path}...")
            with open(file_path, "rb") as file:
                return pickle.load(file)
        config.log.info(f"Data not found at {file_path}. Fetching data...")
        data = func(*args, **kwargs)
        config.log.info(f"Caching data to {file_path}...")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file:
            pickle.dump(data, file)
        config.log.warning(
            "Note that caches are invalidated when either `feature_window_size`"
            " or `target_window_size` are changed."
        )
        return data

    return _cache


def dataframe_to_dataset(df: pd.DataFrame, config: "Config") -> tuple[Tensor, Tensor]:
    """
    Convert a DataFrame to a PyTorch dataset.

    :param df: The DataFrame to convert
    :param config: The configuration object
    """
    X = torch.tensor(df.to_numpy()).float()

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
def alpaca(config: "Config") -> tuple[Tensor, Tensor]:
    time_series = fetch_data(config=config)
    features, targets = indicator_tensors(time_series, config=config)
    return features, targets


@cache
def ett(
    config: "Config", directory: str | None = None, file: str = "ETTh1.csv"
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

    X, y = dataframe_to_dataset(X, config)

    X = X[..., : config.max_features]
    y = y[..., : config.max_features]

    return X, y


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

    X, y = dataframe_to_dataset(X, config)

    X = X[..., : config.max_features]
    y = y[..., : config.max_features]

    return X, y

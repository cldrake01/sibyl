import os
import pickle

import pandas as pd
import torch
from torch import Tensor

from sibyl.utils.log import find_root_dir
from sibyl.utils.preprocessing import indicator_tensors
from sibyl.utils.retrieval import fetch_data


def cache(func):
    def wrapper(*args, **kwargs):
        config = args[0]
        root = find_root_dir(os.path.dirname(__file__))
        file_path = f"{root}/assets/pkl/{func.__name__}.pkl"
        if os.path.exists(file_path):
            config.log.info(f"Loading cached data from {file_path}...")
            with open(file_path, "rb") as f:
                return pickle.load(f)
        else:
            config.log.info(f"Caching data to {file_path}...")
            data = func(*args, **kwargs)
            with open(file_path, "wb") as f:
                pickle.dump(data, f)
            return data

    return wrapper


@cache
def alpaca(config: "Config") -> tuple[Tensor, Tensor]:
    time_series = fetch_data(config=config)
    config.log.info("Creating tensors...")
    features, targets = indicator_tensors(time_series, config=config)
    return features, targets


@cache
def ett(
    config: "Config",
    directory: str = "sibyl/assets/datasets/ETT-small",
    file: str = "ETTh1.csv",
) -> tuple[Tensor, Tensor]:
    """
    Parse the ETT CSVs and return tensors of shape (batch, features, time).

    :return: Feature windows, target windows
    """
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
    directory: str = "/Users/collin/PycharmProjects/sibyl/assets/datasets/ELD",
    file: str = "LD2011_2014.txt",
) -> tuple[Tensor, Tensor]:
    """
    Parse the ELD text file and return tensors of shape (batch, features, time).
    """
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

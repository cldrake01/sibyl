import logging
import os.path
from dataclasses import dataclass
from logging import Logger

import torch
from dotenv import load_dotenv

from sibyl.utils.models.informer.model import Informer
from sibyl.utils.tickers import tickers

load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_WEBSOCKET_KEY = os.getenv("ALPACA_WEBSOCKET_KEY")
ALPACA_WEBSOCKET_SECRET = os.getenv("ALPACA_WEBSOCKET_SECRET")


class NullLogger:
    """
    A logger that does nothing.
    """

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


@dataclass
class TimeSeriesConfig:

    """
    Configuration for time series data.
    """

    years: float = 0.05
    max_workers: int = len(tickers) // 2
    feature_window_size: int = 60
    target_window_size: int = 15
    rate: int = 125
    include_hashes: bool = False
    include_temporal: bool = False
    included_indicators: list[str] = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log: NullLogger | logging.Logger = NullLogger()


@dataclass
class TrainingConfig:
    """
    Configuration for training.
    """

    validation: bool = False
    epochs: int = 10  # Our dataset is quite large, so we don't need many epochs; especially on minute-by-minute data
    batch_size: int = 1
    train_val_split: float = 0.9
    learning_rate: float = 0.001
    criterion: str | torch.nn.modules.loss._Loss = "MSE"
    optimizer: str | torch.optim.Optimizer = "AdamW"
    load_path: str = None
    save_path: str = None
    plot_loss: bool = False
    plot_predictions: bool = False
    plot_interval: int = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log: NullLogger | logging.Logger = NullLogger()

    def __post_init__(self):
        """
        Post-initialization method to set the criterion and optimizer.

        Note that `self.criterion` and `self.optimizer` hold pointers to their respective classes.
        Consequently, they must be instantiated before being used.
        E.g.:
        ```py
        criterion = self.criterion(model.parameters(), ...)
        optimizer = self.optimizer()
        ```
        """
        loss_functions = {
            "MSE": torch.nn.MSELoss,
            "MAE": torch.nn.L1Loss,
        }
        self.criterion = loss_functions[self.criterion]

        optimizers = {
            "Adam": torch.optim.Adam,
            "AdamW": torch.optim.AdamW,
        }
        self.optimizer = optimizers[self.optimizer]


def find_root_dir(current_path, marker_file) -> str:
    """
    Recursively find the root directory by looking for a marker file or directory.
    """
    if os.path.exists(os.path.join(current_path, marker_file)):
        return current_path
    else:
        parent = os.path.dirname(current_path)
        if parent == current_path:
            # Root directory reached without finding the marker
            raise FileNotFoundError(f"Root directory marker '{marker_file}' not found.")
        return find_root_dir(parent, marker_file)


def logger(file_name: str) -> Logger:
    """
    Setup logging configuration

    :param file_name: Path to the log file
    """
    # Identify the root directory based on a marker file
    root_dir = find_root_dir(os.path.dirname(__file__), "README.md")
    log_directory = os.path.join(root_dir, "logs")
    log_file_path = os.path.join(log_directory, f"{file_name}.log")

    # Create a logger
    log = logging.getLogger("my_logger")
    log.setLevel(logging.DEBUG)

    # Create a file handler to log to a log file
    file_handler = logging.FileHandler(log_file_path)

    # Create a formatter to format log messages
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s"
    )

    # Set the formatter for the file handler
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    log.addHandler(file_handler)

    # Send an initial message
    log.info(f"Logging to {file_name}")

    return log

import logging
import os.path
from dataclasses import dataclass

import torch

from sibyl.utils.models.informer.model import Informer

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
WEBSOCKET_KEY = os.getenv("WEBSOCKET_KEY")
WEBSOCKET_SECRET = os.getenv("WEBSOCKET_SECRET")


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

    feature_window_size = 60
    target_window_size = 15
    include_hashes: bool = False
    include_temporal: bool = False
    included_indicators: list[str] = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log: NullLogger or logging.Logger = NullLogger()


@dataclass
class TrainingConfig:
    """
    Configuration for training.
    """

    validation: bool = False
    epochs: int = 10  # Our dataset is quite large, so we don't need many epochs; especially on minute-by-minute data
    batch_size: int = 1
    learning_rate: float = 0.001
    loss_function: str = "MAE"
    optimizer: str = "AdamW"
    load_path: str = None
    save_path: str = None
    plot_loss: bool = False
    plot_predictions: bool = False
    plot_interval: int = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log: NullLogger or logging.Logger = NullLogger()

    def loss_function_(self) -> torch.nn.Module:
        """
        Return the loss function based on the configuration.
        """
        loss_functions = {
            "MSE": torch.nn.MSELoss,
            "MAE": torch.nn.L1Loss,
        }
        return loss_functions[self.loss_function]

    def optimizer_(self) -> torch.optim.Adam or torch.optim.AdamW:
        """
        Return the optimizer based on the configuration.
        """
        optimizers = {
            "Adam": torch.optim.Adam,
            "AdamW": torch.optim.AdamW,
        }
        return optimizers[self.optimizer]


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


def logger(file_name: str) -> logging.Logger:
    """
    Setup logging configuration

    :param file_name: Path to the log file
    """
    # Identify the root directory based on a marker file
    root_dir = find_root_dir(os.path.dirname(__file__), "README.md")
    log_directory = os.path.join(root_dir, "logs")
    log_file_path = os.path.join(log_directory, f"{file_name}.log")

    print(f"Logging to {log_file_path}")

    # Create a logger
    log = logging.getLogger("my_logger")
    log.setLevel(logging.DEBUG)

    # Create a file handler to log to a log file
    file_handler = logging.FileHandler(log_file_path)

    # Create a formatter to format log messages
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Set the formatter for the file handler
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    log.addHandler(file_handler)

    # Send an initial message
    log.info(f"Logging to {file_name}")

    return log

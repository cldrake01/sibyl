import logging
import os.path
from dataclasses import dataclass

from sibyl.utils.models.informer.model import Informer


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
    log: NullLogger or logging.Logger = NullLogger()
    model = Informer


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

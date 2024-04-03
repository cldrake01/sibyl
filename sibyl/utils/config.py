import os
from dataclasses import dataclass
from logging import Logger

import torch
from torch import Tensor

from sibyl import tickers
from sibyl.utils.datasets import alpaca, ett, eld
from sibyl.utils.log import NullLogger, logger
from sibyl.utils.loss import MaxSE, MaxAE, MaxAPE, Fourier, CMaxSE, CMaxAE, WaveletLoss


@dataclass
class Config:
    """
    Configuration for training.
    """

    years: float = 0.05
    max_workers: int = len(tickers) // 2
    feature_window_size: int = 60
    target_window_size: int = 15
    rate: int = 125
    include_hashes: bool = False
    include_temporal: bool = False
    included_indicators: list[str] | None = None
    validation: bool = True
    epochs: int = 10
    batch_size: int = 1
    train_val_split: float = 0.9
    learning_rate: float = 0.001
    criterion: str | torch.nn.Module = "MSE"
    optimizer: str | torch.optim.Optimizer = "AdamW"
    load_path: str | None = None
    save_path: str | None = None
    plot_loss: bool = False
    plot_predictions: bool = False
    plot_interval: int = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name: str = "alpaca"
    dataset: tuple[Tensor, Tensor] | None = None
    log: NullLogger | Logger = NullLogger()
    logger_name: str = ""

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
        # Check for macOS and set environment variable to avoid MKL errors
        if os.name == "posix":
            os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

        loss_functions = {
            "Fourier": Fourier,
            "MaxAE": MaxAE,
            "MaxSE": MaxSE,
            "MaxAPE": MaxAPE,
            "MSE": torch.nn.MSELoss,
            "MAE": torch.nn.L1Loss,
            "CMaxSE": CMaxSE,
            "CMaxAE": CMaxAE,
            "WaveletLoss": WaveletLoss,
        }
        self.criterion = loss_functions[self.criterion]

        optimizers = {
            "Adam": torch.optim.Adam,
            "AdamW": torch.optim.AdamW,
        }
        self.optimizer = optimizers[self.optimizer]

        if self.logger_name:
            self.log = logger(self.logger_name, self.dataset_name)

        datasets = {
            "alpaca": alpaca,
            "ett": ett,
            "eld": eld,
        }
        self.dataset = datasets[self.dataset_name](self)

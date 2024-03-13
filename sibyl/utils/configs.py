from dataclasses import dataclass
from logging import Logger

import torch

from sibyl import tickers, MaxAE
from sibyl.utils.log import NullLogger
from sibyl.utils.loss import MaxSE


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
    log: NullLogger | Logger = NullLogger()


@dataclass
class TrainingConfig:
    """
    Configuration for training.
    """

    validation: bool = False
    # Our dataset is quite large, so we don't need many epochs; especially on minute-by-minute data
    epochs: int = 10
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
    log: NullLogger | Logger = NullLogger()

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
            "MaxAE": MaxAE,
            "MaxSE": MaxSE,
            "MSE": torch.nn.MSELoss,
            "MAE": torch.nn.L1Loss,
        }
        self.criterion = loss_functions[self.criterion]

        optimizers = {
            "Adam": torch.optim.Adam,
            "AdamW": torch.optim.AdamW,
        }
        self.optimizer = optimizers[self.optimizer]

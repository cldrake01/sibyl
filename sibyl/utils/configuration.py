import os
from dataclasses import dataclass
from logging import Logger
from typing import Any

import torch
from torch import Tensor, nn

from sibyl import tickers, Informer
from sibyl.utils.datasets import alpaca, ett, eld
from sibyl.utils.logging import NullLogger, Log
from sibyl.utils.loss import (
    VMaxSE,
    VMaxAE,
)
from sibyl.utils.models.dimformer.model import Dimformer
from sibyl.utils.models.informer.model import DecoderOnlyInformer
from sibyl.utils.models.ring.ring_attention import RingTransformer


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
    epoch: int = 0
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
    metrics: dict[str, list[float]] | None = None
    stage: str = "Preprocessing"

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

        self.metrics = self.metrics or {}

        loss_functions = {
            # "Fourier": Fourier,
            "VMaxAE": VMaxAE,
            "VMaxSE": VMaxSE,
            # "MaxAPE": MaxAPE,
            "MSE": torch.nn.MSELoss,
            "MAE": torch.nn.L1Loss,
            # "CMaxSE": CMaxSE,
            # "CMaxAE": CMaxAE,
            # "WaveletLoss": Wave,
        }
        self.criterion = loss_functions[self.criterion]

        optimizers = {
            "Adam": torch.optim.Adam,
            "AdamW": torch.optim.AdamW,
        }
        self.optimizer = optimizers[self.optimizer]

        if self.logger_name:
            self.log = Log(self.logger_name, self.dataset_name).logger

        datasets = {
            "alpaca": alpaca,
            "ett": ett,
            "eld": eld,
        }
        self.dataset = datasets[self.dataset_name](self)


def initialize_model(X: Tensor, y: Tensor, model: Any) -> nn.Module:
    """
    Initialize the model based on the configuration.

    :param X: The features.
    :param y: The targets.
    :param model: The model to initialize.
    """
    num_features = X.size(2)
    num_targets = y.size(2)
    feature_len = X.size(1)
    target_len = y.size(1)

    model_configurations = {
        Dimformer: Dimformer(
            enc_in=num_features,
            dec_in=num_features,
            # c_out=num_features,
            c_out=target_len,
            seq_len=feature_len,
            label_len=target_len,
            out_len=target_len,
            factor=5,
            d_model=512,
            n_heads=num_features,
            e_layers=3,
            d_layers=2,
            d_ff=512,
            dropout=0.05,
            attn="self",
            embed="fixed",
            freq="h",
            activation="gelu",
            output_attention=False,
            distil=False,
            mix=True,
            encoder=False,
        ),
        Informer: Informer(
            enc_in=num_features,
            dec_in=num_features,
            c_out=num_features,
            seq_len=feature_len,
            label_len=target_len,
            out_len=target_len,
            factor=5,
            d_model=512,
            n_heads=num_features,
            e_layers=3,
            d_layers=2,
            d_ff=512,
            dropout=0.05,
            attn="prob",
            embed="fixed",
            freq="h",
            activation="gelu",
            output_attention=False,
            distil=True,
            mix=True,
        ),
        DecoderOnlyInformer: DecoderOnlyInformer(
            dec_in=num_features,
            c_out=num_targets,
            seq_len=feature_len,
            out_len=target_len,
            factor=5,
            d_model=512,
            n_heads=num_features,
            d_layers=2,
            d_ff=512,
            dropout=0.01,
            activation="gelu",
        ),
        RingTransformer: Ring(
            enc_in=num_features,
            dec_in=num_features,
            c_out=num_targets,
            seq_len=feature_len,
            label_len=target_len,
            out_len=target_len,
            factor=5,
            d_model=512,
            n_heads=num_features,
            e_layers=3,
            d_layers=2,
            d_ff=512,
            dropout=0.01,
            attn="prob",
            embed="fixed",
            freq="h",
            activation="gelu",
            output_attention=False,
            distil=True,
            mix=True,
        ),
    }

    return model_configurations[model]
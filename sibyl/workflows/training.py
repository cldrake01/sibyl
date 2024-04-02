import os
import signal
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm

from sibyl.utils.config import Config
from sibyl.utils.log import find_root_dir
from sibyl.utils.loss import bias_variance_decomposition, MaxAPE
from sibyl.utils.models.dimformer.model import Dimformer
from sibyl.utils.models.informer.model import Informer, DecoderOnlyInformer
from sibyl.utils.models.ring.model import Ring
from sibyl.utils.models.ring.ring_attention import RingTransformer
from sibyl.utils.plot import pred_plot, bias_variance_plot
from sibyl.utils.preprocessing import normalize


def initialize_model(X: Tensor, y: Tensor, model: Any) -> nn.Module:
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


def prepare_datasets(
    X: Tensor, y: Tensor, config: Config
) -> tuple[DataLoader, DataLoader]:
    total_samples = len(X)
    train_size = int(total_samples * config.train_val_split)
    val_size = total_samples - train_size
    full_dataset = TensorDataset(X, y)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=False
    )
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    return train_loader, val_loader


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Config,
):
    config.criterion = config.criterion()
    config.optimizer = config.optimizer(model.parameters(), lr=config.learning_rate)

    for param in model.parameters():
        param.requires_grad = True

    # Define a function to save the model
    def save_model(model: nn.Module, filepath: str):
        config.log.info("Saving model...")
        torch.save(model.state_dict(), filepath)
        config.log.info("Model saved successfully.")

    # Register a signal handler to save the model upon termination
    def signal_handler(sig, frame):
        config.log.info("Program interrupted.")
        save_model(
            model,
            f"{find_root_dir(os.path.dirname(__file__))}/assets/models/{config.dataset_name}-model.pt",
        )
        exit()

    signal.signal(signal.SIGINT, signal_handler)

    # maxape = MaxAPE(benchmark=True)

    for epoch in range(config.epochs):
        model.train()
        training_losses: list[float] = []
        mae, mse, rs = [], [], []
        bias_variance, bias, variance = [], [], []
        train_loss = 0.0  # Reset train loss for the epoch

        for window, (X, y) in enumerate(tqdm(train_loader, desc="Training")):
            config.optimizer.zero_grad()
            y_hat = model(X, y)
            loss = config.criterion(y_hat, y)
            loss.backward()
            config.optimizer.step()
            train_loss += loss.item()

            training_losses.append(loss.item())
            # mae.append(torch.nn.functional.l1_loss(y_hat, y).item())
            # mse.append(torch.nn.functional.mse_loss(y_hat, y).item())
            rs.append(torch.sum(torch.abs(y - y_hat)).item())

            b_v = bias_variance_decomposition(y_hat, y)
            bias_variance.append(b_v[0])
            bias.append(b_v[1])
            variance.append(b_v[2])

            if window == 20_000:
                bias_variance_plot(
                    bias_variance,
                    bias,
                    variance,
                    rs,
                    config,
                )
                return

            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            if window % config.plot_interval == 0:
                pred_plot(
                    X=X,
                    y=y,
                    y_hat=y_hat,
                    loss=training_losses,
                    config=config,
                )

        model.eval()
        validation_losses: list[float] = []
        val_loss = 0.0  # Reset validation loss for the epoch

        with torch.no_grad():
            for window, (X, y) in enumerate(tqdm(val_loader, desc="Validating")):
                y_hat = model(X, y)
                loss = config.criterion(y_hat, y)
                val_loss += loss.item()
                validation_losses.append(loss.item())
                if window % config.plot_interval == 0:
                    pred_plot(
                        X=X,
                        y=y,
                        y_hat=y_hat,
                        loss=validation_losses,
                        config=config,
                    )
    config.log.info("Training complete.")
    save_model(model, f"{find_root_dir(os.path.dirname(__file__))}/assets/model.pt")


def main():
    """
    Having a main function allows us to run the script within a Jupyter Notebook.

    ```py
    from sibyl.workflows.training import main

    main()
    ```

    You can, for example, import your Sibyl fork from a private GitHub repository as a package and run the main
    function. You must override `setup.py` if you intend to utilize Sibyl as a custom package.
    """
    loss_functions = (
        # "Fourier",
        "MaxAE",
        "MaxSE",
        # "MaxAPE",
        "MSE",
        "MAE",
        # "CMaxSE",
        # "CMaxAE",
    )

    for loss in loss_functions:
        config = Config(
            epochs=10,
            learning_rate=0.001,
            criterion=loss,
            optimizer="AdamW",
            plot_loss=False,
            plot_predictions=False,
            plot_interval=300,
            dataset_name="alpaca",
            feature_window_size=120,
            target_window_size=15,
            include_hashes=False,
            include_temporal=False,
            included_indicators=[
                "ROC",
                "RSI",
                # "MFI",
                "ADX",
            ],
            years=0.01,
            logger_name=os.path.basename(__file__),
        )
        features, targets = config.dataset
        X, y = normalize(features, targets)
        model = initialize_model(X, y, Dimformer)
        train_loader, val_loader = prepare_datasets(X, y, config)
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
        )


if __name__ == "__main__":
    main()

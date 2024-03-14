import os
import pickle
import signal

import torch
from torch import Tensor, nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm

from sibyl.utils.configs import TimeSeriesConfig, TrainingConfig
from sibyl.utils.log import logger, find_root_dir
from sibyl.utils.models.dimformer.model import Dimformer
from sibyl.utils.models.informer.model import Informer, DecoderOnlyInformer
from sibyl.utils.models.ring.model import Ring
from sibyl.utils.models.ring.ring_attention import RingTransformer
from sibyl.utils.plot import plot
from sibyl.utils.preprocessing import indicator_tensors, normalize
from sibyl.utils.retrieval import fetch_data


def setup_environment():
    # Check for macOS
    if os.name == "posix":
        os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    return logger("training.py")


def load_and_preprocess_data(
    config: TimeSeriesConfig, file_path: str | None = None
) -> tuple[Tensor, Tensor]:
    root = find_root_dir(os.path.dirname(__file__))

    file_path = file_path or f"{root}/assets/pkl/time_series.pkl"

    if os.path.exists(file_path):
        config.log.info("Loading pickle file...")
        with open(file_path, "rb") as f:
            time_series = pickle.load(f)
    else:
        time_series = fetch_data(config=config)
        config.log.info("Creating pickle file...")
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        with open(file_path, "wb") as f:
            pickle.dump(time_series, f)

    config.log.info("Creating tensors...")
    features, targets = indicator_tensors(time_series, config=config)
    return features, targets


def initialize_model(X: Tensor, y: Tensor, model: type) -> torch.nn.Module:
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
            distil=True,
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
    X: Tensor, y: Tensor, config: TrainingConfig
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
    config: TrainingConfig,
):
    config.criterion = config.criterion()
    config.optimizer = config.optimizer(model.parameters(), lr=config.learning_rate)

    # Define a function to save the model
    def save_model(model: nn.Module, filepath: str):
        config.log.info("Saving model...")
        torch.save(model.state_dict(), filepath)
        config.log.info("Model saved successfully.")

    # Register a signal handler to save the model upon termination
    def signal_handler(sig, frame):
        config.log.info("Program interrupted.")
        save_model(
            model, f"{find_root_dir(os.path.dirname(__file__))}/assets/models/model.pt"
        )
        exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    for epoch in range(config.epochs):
        model.train()
        training_losses: list = []
        train_loss = 0.0  # Reset train loss for the epoch

        for window, (X, y) in enumerate(tqdm(train_loader, desc="Training")):
            config.optimizer.zero_grad()
            y_hat = model(X, y)
            loss = config.criterion(y_hat, y)
            if window == 20_000:
                return
            loss.backward()
            config.optimizer.step()
            train_loss += loss.item()
            training_losses.append(loss.item())
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            if window % config.plot_interval == 0:
                plot(
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
                    plot(
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
    from import sibyl.workflows.training import main

    main()
    ```

    You can, for example, import your Sibyl fork from a private GitHub repository as a package and run the main
    function. You must override `setup.py` if you intend to utilize Sibyl as a custom package.
    """
    log = setup_environment()
    time_series_config = TimeSeriesConfig(
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
        log=log,
        years=0.005,
    )
    features, targets = load_and_preprocess_data(time_series_config)
    X_norm, y_norm = normalize(features, targets)
    model = initialize_model(X_norm, y_norm, Dimformer)
    training_config = TrainingConfig(
        validation=True,
        epochs=10,
        learning_rate=0.001,
        criterion="MaxAE",
        optimizer="AdamW",
        plot_loss=True,
        plot_predictions=True,
        plot_interval=300,
        log=log,
    )
    train_loader, val_loader = prepare_datasets(X_norm, y_norm, training_config)
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
    )


if __name__ == "__main__":
    main()

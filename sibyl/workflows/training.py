import os
import signal
from typing import Generator

import torch
from torch import Tensor, nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm

from sibyl.utils.benchmarking import stats, bias, variance
from sibyl.utils.configuration import Config, initialize_model
from sibyl.utils.logging import find_root_dir
from sibyl.utils.loss import VMaxSE, VMaxAE
from sibyl.utils.models.dimformer.model import Dimformer
from sibyl.utils.plotting import predicted_vs_actual, metrics_table, metrics
from sibyl.utils.preprocessing import normalize


def prepare_datasets(
    X: Tensor, y: Tensor, config: Config
) -> tuple[DataLoader, DataLoader]:
    """
    Prepare the training and validation datasets.

    :param X: The features.
    :param y: The targets.
    :param config: The configuration object.
    """
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


@stats(bias, variance)
def train(
    model: nn.Module, loader: DataLoader, config: Config
) -> Generator[tuple[Tensor, Tensor], None, None]:
    """
    Train the model using the training dataset.

    :param model: The model to train.
    :param loader: The training dataset.
    :param config: The configuration object.
    """
    config.stage = "Training"
    model.train()
    losses: list[float] = []
    train_loss = 0.0  # Reset train loss for the epoch

    for step, (X, y) in enumerate(tqdm(loader, desc="Training")):
        config.optimizer.zero_grad()
        y_hat = model(X, y)
        loss = config.criterion(y_hat, y)
        loss.backward()
        config.optimizer.step()
        train_loss += loss.item()
        losses.append(loss.item())
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        if step % config.plot_interval == 0:
            predicted_vs_actual(
                X=X,
                y=y,
                y_hat=y_hat,
                loss=losses,
                config=config,
            )
        yield y, y_hat


@stats(bias, variance, VMaxSE(benchmark=True), VMaxAE(benchmark=True))
def validate(
    model: nn.Module, loader: DataLoader, config: Config
) -> Generator[tuple[Tensor, Tensor], None, None]:
    """
    Validate the model using the validation dataset.

    :param model: The model to validate.
    :param loader: The validation dataset.
    :param config: The configuration object.
    """
    config.stage = "Validation"
    model.eval()
    losses: list[float] = []
    val_loss = 0.0

    with torch.no_grad():
        for step, (X, y) in enumerate(tqdm(loader, desc="Validating")):
            y_hat = model(X, y)
            loss = config.criterion(y_hat, y)
            val_loss += loss.item()
            losses.append(loss.item())
            if step % config.plot_interval == 0:
                predicted_vs_actual(
                    X=X,
                    y=y,
                    y_hat=y_hat,
                    loss=losses,
                    config=config,
                )
            yield y, y_hat


def build_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Config,
):
    """
    Train the model using the training and validation datasets.

    :param model: The model to train.
    :param train_loader: The training dataset.
    :param val_loader: The validation dataset.
    :param config: The configuration object.
    """
    config.criterion = config.criterion()
    config.optimizer = config.optimizer(model.parameters(), lr=config.learning_rate)

    # Define a function to save the model
    def save_model(model_: nn.Module, filepath: str):
        config.log.info("Saving model...")
        torch.save(model_.state_dict(), filepath)
        config.log.info("Model saved successfully.")

    # Register a signal handler to save the model upon termination
    def signal_handler(sig, frame):
        config.log.info("Program interrupted.")
        path_ = find_root_dir(os.path.dirname(__file__))
        path_ += f"/assets/models/{config.dataset_name}-model.pt"
        save_model(model, path_)

    signal.signal(signal.SIGINT, signal_handler)

    for epoch in range(config.epochs):
        config.epoch = epoch

        train(model, train_loader, config)

        metrics(config)

        metrics_table(config)

        validate(model, val_loader, config)

        metrics(config)

        metrics_table(config)

    config.log.info("Training complete.")
    path = find_root_dir(os.path.dirname(__file__))
    path += f"/assets/models/{config.dataset_name}-model.pt"
    save_model(model, path)


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

    aggregated_metrics = []
    loss_functions = ["VMaxAE", "VMaxSE", "MSE", "MAE"]

    for loss in loss_functions:
        config = Config(
            epochs=1,
            learning_rate=0.001,
            criterion=str(loss),
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
                "ADX",
            ],
            years=0.0009,
            logger_name=os.path.basename(__file__),
        )
        features, targets = config.dataset
        X, y = normalize(features, targets)
        model = initialize_model(X, y, Dimformer)
        train_loader, val_loader = prepare_datasets(X, y, config)
        build_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
        )
        aggregated_metrics.append((loss, config.metrics))


if __name__ == "__main__":
    main()

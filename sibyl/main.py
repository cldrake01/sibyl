import os
import signal
import time
from typing import Generator

import torch
from torch import Tensor, nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm

from sibyl.utils.benchmarking import stats, bias, variance, std
from sibyl.utils.configuration import Config, initialize_model
from sibyl.utils.logging import find_root_dir
from sibyl.utils.loss import VMaxSE, VMaxAE
from sibyl.utils.models.dimformer.model import Dimformer
from sibyl.utils.plotting import predicted_vs_actual, metrics_table, metrics
from sibyl.utils.preprocessing import normalize


def prepare_datasets(
    X: Tensor,
    Y: Tensor,
    config: Config,
) -> tuple[DataLoader, DataLoader]:
    """
    Prepare the training and validation datasets.

    :param X: Features of shape (batch, time, features).
    :param Y: Targets of shape (batch, time, features).
    :param config: The configuration object.
    """
    total_samples = len(X)
    train_size = int(total_samples * config.train_val_split)
    val_size = total_samples - train_size
    full_dataset = TensorDataset(X, Y)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=False
    )
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    return train_loader, val_loader


@stats(bias, variance, std, VMaxSE.mse, VMaxAE.mae)
def train(
    model: nn.Module,
    loader: DataLoader,
    config: Config,
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
    best_loss = float("inf")
    best_x, best_y, best_y_hat = None, None, None

    for step, (x, y) in enumerate(tqdm(loader, desc="Training")):
        config.optimizer.zero_grad()
        # y_hat.size() = [batch, predicted_len, features]
        y_hat = model(x, y)
        loss = config.criterion(y_hat, y)
        loss.backward()
        config.optimizer.step()
        train_loss += loss.item()
        losses.append(loss.item())
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        if loss < best_loss:
            best_loss = loss.item()
            best_x, best_y, best_y_hat = x, y, y_hat
        if config.plot_interval and step % config.plot_interval == 0:
            predicted_vs_actual(
                x=x,
                y=y,
                y_hat=y_hat,
                loss=losses,
                config=config,
            )
            # residuals(
            #     y=y,
            #     y_hat=y_hat,
            #     features=None,
            #     config=config,
            # )
        yield y, y_hat

    name = config.criterion.__class__.__name__
    config.log.metric(f"Best training loss: {best_loss:.5f} {name}")
    predicted_vs_actual(
        x=best_x,
        y=best_y,
        y_hat=best_y_hat,
        loss=losses,
        config=config,
        filename=f"best-{config.criterion.__class__.__name__}",
    )


@stats(bias, variance, std)
def validate(
    model: nn.Module,
    loader: DataLoader,
    config: Config,
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
    best_loss = float("inf")
    best_x, best_y, best_y_hat = None, None, None

    with torch.no_grad():
        for step, (x, y) in enumerate(tqdm(loader, desc="Validating")):
            y_hat = model(x, y)
            loss = config.criterion(y_hat, y)
            val_loss += loss.item()
            losses.append(loss.item())
            if loss < best_loss:
                best_loss = loss.item()
                best_x, best_y, best_y_hat = x, y, y_hat
            if config.plot_interval and step % config.plot_interval == 0:
                predicted_vs_actual(
                    x=x,
                    y=y,
                    y_hat=y_hat,
                    loss=losses,
                    config=config,
                )
            yield y, y_hat

    config.log.metric(f"Best validation loss: {best_loss:.5f}")
    predicted_vs_actual(
        x=best_x,
        y=best_y,
        y_hat=best_y_hat,
        loss=losses,
        config=config,
        filename=f"best-{config.criterion.__class__.__name__}",
    )


def build_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Config,
) -> None:
    """
    Train the model using the training and validation datasets.

    :param model: The model to train.
    :param train_loader: The training dataset.
    :param val_loader: The validation dataset.
    :param config: The configuration object.
    """
    start = time.perf_counter_ns()

    config.criterion = config.criterion()
    config.optimizer = config.optimizer(model.parameters(), lr=config.learning_rate)

    # Define a function to save the model
    def save_model(model_: nn.Module, filepath: str):
        config.log.info("Saving model...")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
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

        validate(model, val_loader, config)

    config.log.info("Training complete.")
    config.log.info(
        f"Total time elapsed: {(time.perf_counter_ns() - start) / 1e9:.2f} seconds."
    )
    path = find_root_dir(os.path.dirname(__file__))
    path += f"/assets/models/{config.dataset_name}-model.pt"
    save_model(model, path)


def main() -> None:
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
    loss_functions = [
        "VMaxSE",
        "MSE",
        "VMaxAE",
        "MAE",
    ]
    # loss_functions = ["VMaxSE"]

    # "alpaca", "ett", or "eld"
    dataset = "alpaca"

    for loss in loss_functions:
        config = Config(
            epochs=1,
            learning_rate=0.001,
            criterion=loss,
            optimizer="AdamW",
            plot_loss=True,
            plot_predictions=True,
            plot_interval=10_000,
            dataset_name=dataset,
            X_window_size=120,
            Y_window_size=15,
            included_indicators=[
                "ROC",
                "RSI",
                "ADX",
            ],
            years=0.0027,  # 1 day
            features=3,
            batches=30_000,
            log_file_name=os.path.basename(__file__),
        )
        X, Y = config.__dataset
        X, Y = normalize(X, Y)
        model = initialize_model(X, Y, Dimformer)
        train_loader, val_loader = prepare_datasets(X, Y, config)
        build_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
        )
        aggregated_metrics.append((loss, config.metrics))

    metrics_table(aggregated_metrics, dataset)


if __name__ == "__main__":
    main()

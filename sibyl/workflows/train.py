import os
import pickle

import torch
from IPython.core.display_functions import clear_output
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader, random_split

from sibyl import logger, find_root_dir, NullLogger, TimeSeriesConfig
from sibyl.utils.models.informer.model import Informer
from sibyl.utils.preprocessing import indicator_tensors
from sibyl.utils.retrieval import fetch_data


def setup_environment():
    # Check for macOS
    if os.name == "posix":
        os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    return logger("train.py"), find_root_dir(os.path.dirname(__file__), "README.md")


def load_and_preprocess_data(log, file_path=None) -> tuple[torch.Tensor, torch.Tensor]:
    if file_path is None:
        file_path = f"{find_root_dir(os.path.dirname(__file__), 'README.md')}/assets/pkl/time_series.pkl"

    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            time_series = pickle.load(f)
    else:
        time_series = fetch_data(years=0.05, log=log)
        log.info("Creating pickle file...")
        with open(file_path, "wb") as f:
            pickle.dump(time_series, f)

    config = TimeSeriesConfig(
        include_hashes=True,
        include_temporal=True,
        log=log,
    )

    log.info("Creating tensors...")
    features, targets = indicator_tensors(time_series, config=config)
    log.info("Tensors created.")
    return features, targets


def initialize_model(X, y):
    num_features = X.size(2)
    num_targets = y.size(2)
    feature_len = X.size(1)
    target_len = y.size(1)

    # return DecoderOnlyInformer(
    #     dec_in=num_features,
    #     c_out=num_targets,
    #     seq_len=feature_len,
    #     out_len=target_len,
    #     factor=5,
    #     d_model=512,
    #     n_heads=num_features,
    #     d_layers=2,
    #     d_ff=512,
    #     dropout=0.01,
    #     activation="gelu",
    # )

    return Informer(
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
    )


def normalize(X, y) -> tuple[torch.Tensor, torch.Tensor]:
    """
    See \frac{\left|T\right|}{T}\log_{10}\left(\left|T\right|+1\right), where T is the tensor
    to be normalized.

    This normalization preserves the sign of the tensor whilst normalizing it, as opposed to
    methods available online, wherein the minimum is added to the tensor before normalizing it;
    thereby shifting the tensor to the positive side of the number line, subsequently losing the
    sign of the tensor.

    :param X: The feature tensor.
    :param y: The target tensor.
    :return: The normalized feature and target tensors.
    """
    X = torch.nan_to_num(torch.sign(X) * torch.log10(torch.abs(X) + 1.0).float(), 0.0)
    y = torch.nan_to_num(torch.sign(y) * torch.log10(torch.abs(y) + 1.0).float(), 0.0)

    return X, y


def prepare_datasets(
    X, y, train_size_ratio=0.9, batch_size=1
) -> tuple[DataLoader, DataLoader]:
    total_samples = len(X)
    train_size = int(total_samples * train_size_ratio)
    val_size = total_samples - train_size
    full_dataset = TensorDataset(X, y)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def early_stop_check(
    loss, best_loss, patience, current_patience, model, log=NullLogger()
):
    if best_loss is None:
        best_loss = loss

    if loss < best_loss:
        best_loss = loss
        current_patience = 0
    else:
        current_patience += 1

    if current_patience >= patience:
        log.info(f"Early stopping triggered. Best loss: {best_loss}")
        save_model(
            model,
            f"{find_root_dir(os.path.dirname(__file__), 'README.md')}/assets/weights/informer.pt",
        )
        exit(0)

    # log.info(f"Loss: {loss} | Best Loss: {best_loss} | Patience: {current_patience}")


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    log=NullLogger(),
    validation=False,
    epochs=1,  # Default to 1 epoch if not specified
):
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters())
    best_loss = float("inf")
    early_stopping_patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        log.info(f"Epoch: {epoch + 1}/{epochs}")
        model.train()
        training_losses = []
        train_loss = 0.0  # Reset train loss for the epoch

        for window, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = model(X, y)
            clear_output(wait=True)
            # log.info(f"y_hat: {y_hat[0][-1]}")
            # log.info(f"y: {y[0][-1]}")
            # log.info(f"y_hat - y: {y_hat[0][-1] - y[0][-1]}")
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            training_losses.append(loss.item())

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            # Plot every 20 windows
            if window % 20 == 0:
                pred_vs_actual_plot(X, y, y_hat)

                # live_plot(
                #     loss=training_losses,
                #     title="Training Losses",
                #     window=window,
                #     windows=len(train_loader),
                # )

            # Early stopping check after each window
            early_stop_check(
                loss=train_loss / (window + 1),  # Current average loss
                patience=early_stopping_patience,
                current_patience=patience_counter,
                best_loss=best_loss,
                model=model,
                log=log,
            )

        if validation:
            model.eval()
            validation_losses = []
            val_loss = 0.0  # Reset validation loss for the epoch
            with torch.no_grad():
                for window, (X, y) in enumerate(val_loader):
                    X, y = X.to(device), y.to(device)
                    y_hat = model(X, y)
                    loss = criterion(y_hat, y)
                    val_loss += loss.item()
                    validation_losses.append(loss.item())

                    # # Plot every 20 windows
                    # if window % 20 == 0:
                    #     live_plot(
                    #         loss=validation_losses,
                    #         title="Validation Losses",
                    #         window=window,
                    #         windows=len(val_loader),
                    #     )

            # Update best_loss and reset patience_counter if validation loss improved
            average_val_loss = val_loss / len(val_loader)
            if average_val_loss < best_loss:
                best_loss = average_val_loss
                patience_counter = 0
                # Save the model
                torch.save(model.state_dict(), "best_model.pth")
                log.info(f"New best loss: {best_loss}")
            else:
                patience_counter += 1
                log.info(
                    f"No improvement in validation loss for {patience_counter} epochs."
                )
                if patience_counter >= early_stopping_patience:
                    log.info("Early stopping triggered")
                    return  # Exit the training function


def save_model(model, path=None):
    if path is None:
        path = f"{find_root_dir(os.path.dirname(__file__), 'README.md')}/assets/weights/informer.pt"
    torch.save(model.state_dict(), path)


def pred_vs_actual_plot(X, y, y_hat, feature: int = 5):
    """
    Plot the predicted vs actual values such that X is placed to the left of y and y_hat.

    :param X: The context window preceding the target window.
    :param y: The target window.
    :param y_hat: The predicted target window.
    :param feature: The feature to plot.
    :return:
    """
    # Detach the tensors from the computational graph
    X_, y_, y_hat_ = X.detach(), y.detach(), y_hat.detach()

    # Squeeze the batch dimension
    X_, y_, y_hat_ = X_.squeeze(0), y_.squeeze(0), y_hat_.squeeze(0)

    # Plot the selected feature
    X_, y_, y_hat_ = X_[:, feature], y_[:, feature], y_hat_[:, feature]

    # print(X_.shape, y_.shape, y_hat_.shape)

    # Plot the tensors
    plt.plot(torch.cat((X_, y_)), "b", alpha=0.5)
    plt.plot(torch.cat((X_, y_hat_)), "r", alpha=0.5)
    plt.show()


def live_plot(loss, title="Losses", window=None, windows=None):
    clear_output(wait=True)
    # Destroy the current figure
    plt.clf()
    plt.plot(loss)
    plt.title(title)
    plt.xlabel(
        f"Windows: {window}/{windows}, Mean Loss: {(sum(fifty := loss[:-50]) / len(fifty)) if len(loss) > 60 else 'N/A'}"
    )
    plt.ylabel("Loss")
    plt.show()


def plot_losses(losses):
    """
    Plot losses independently of the current phase, whether training or validation.

    :param losses: A list of losses to plot.
    :return:
    """
    plt.plot(losses)
    plt.show()


def main():
    log, root = setup_environment()
    features, targets = load_and_preprocess_data(log)
    X_norm, y_norm = normalize(features, targets)
    model = initialize_model(X_norm, y_norm)
    train_loader, val_loader = prepare_datasets(X_norm, y_norm)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=1,
        log=log,
    )
    save_model(model)
    # plot_losses logic here


if __name__ == "__main__":
    main()

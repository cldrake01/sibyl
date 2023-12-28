import os
import pickle

import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import TensorDataset, random_split, DataLoader
from tqdm import tqdm

from src import NullLogger, logger
from src.utils.models.model import Informer
from src.utils.preprocessing import stock_tensors
from src.utils.retrieval import fetch_data

log = logger("training.py")

log.info("Checking for pickle file...")

path = "../../assets/pkl/time_series.pkl"

if not os.path.exists(path):
    log.info("Pickle file not found. Fetching data...")
    time_series = fetch_data(years=0.1, log=log)
    log.info("Creating pickle file...")
    with open(path, "wb") as f:
        pickle.dump(time_series, f)
else:
    log.info("Pickle file found.")
    log.info("Loading pickle file...")
    with open(path, "rb") as f:
        time_series = pickle.load(f)

log.info("Creating tensors...")
X, y = stock_tensors(time_series)
log.info("Tensors created.")

model = Informer(
    enc_in=8,
    dec_in=8,
    c_out=8,
    seq_len=60,
    label_len=15,
    out_len=15,
    factor=5,
    d_model=512,
    n_heads=8,
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
    device=torch.device("cuda:0"),
)

# Train the model
log.info("Starting training...")


def train(
    X: torch.Tensor,  # Feature windows tensor
    y: torch.Tensor,  # Target windows tensor
    model,
    normalized: bool = True,
    epochs: int = 1_000_000,
    log=NullLogger(),
    batch_size: int = 1,
) -> None:
    """
    Train the Time Series Transformer model.

    :param X: (torch.Tensor): Feature windows tensor.
    :param y: (torch.Tensor): Target windows tensor.
    :param model: (TimeSeriesTransformer): Your TimeSeriesTransformer model.
    :param normalized: (bool): If True, normalizes the input data.
    :param epochs: (int): Number of epochs to train for.
    :param log: (logging.Logger): Logger (optional).
    :param batch_size: (int): Batch size.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info(f"X.shape: {X.shape}, y.shape: {y.shape}")

    # Generate dummy past_time_features and past_observed_mask
    # num_samples, num_time_steps, num_features = X.shape
    #
    # time_steps_tensor = torch.arange(1, num_time_steps + 1).view(1, num_time_steps, 1)
    # past_time_features = (
    #     time_steps_tensor.expand(-1, -1, num_features).float().to(device)
    # )
    # past_observed_mask = torch.ones(num_time_steps, num_features).float().to(device)

    # log.info(f"past_time_features.shape: {past_time_features.size()}")
    # log.info(f"past_observed_mask.shape: {past_observed_mask.size()}")

    # Normalizing data if required
    if normalized:
        X = torch.log10(X.detach().clone() + 1.0).float()
        y = torch.log10(y.detach().clone() + 1.0).float()

    # Creating a TensorDataset and DataLoader
    log.info("Creating a TensorDataset and DataLoader...")
    dataset = TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    log.info("TensorDataset and DataLoader created.")

    # Model, loss function, and optimizer
    log.info("Moving model to device...")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    log.info("Model moved to device.")

    # Training loop
    best_loss = float("inf")
    early_stopping_patience = 10
    patience_counter = 0

    epoch_train_losses, epoch_val_losses = [], []

    for epoch in tqdm(range(epochs), desc="Training"):
        model.train()
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.float().to(device), y_batch.float().to(device)
            optimizer.zero_grad()

            # log.info(f"X_batch.shape: {X_batch.size()}")
            # log.info(f"y_batch.shape: {y_batch.size()}")

            output = model(
                X_batch, y_batch[:, :-1, :]
            )  # Exclude last time step from target for input
            y_true = y_batch[
                :, 1:, :
            ]  # Exclude first time step from target for loss calculation

            loss = criterion(output, y_true)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation step
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                val_output = model(X_val, y_val[:, :-1, :])
                y_true_val = y_val[:, 1:, :]

                val_loss = criterion(val_output, y_true_val)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        # Early stopping check
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                log.info("Early stopping triggered.")
                break

        # Record the average loss of this epoch
        epoch_train_losses.append(avg_train_loss)
        epoch_val_losses.append(avg_val_loss)

        log.info(
            f"Epoch {epoch}: Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}"
        )

    # Save the model
    if not os.path.exists("models"):
        os.mkdir("models")
    torch.save(model.state_dict(), "models/transformer.pt")
    log.info("Model saved.")

    # Plot loss
    if not os.path.exists("plots"):
        os.mkdir("plots")
    plt.plot(epoch_train_losses, label="Train Loss")
    plt.plot(epoch_val_losses, label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig("plots/loss.png")
    plt.clf()


train(X=X, y=y, model=model, epochs=2_000, log=log)

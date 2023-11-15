import os
import torch
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.nn as nn
import seaborn as sns
import torch.optim as optim
import matplotlib.pyplot as plt
from alpaca.data.historical import *
from alpaca.data import StockBarsRequest, TimeFrame
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from log import setup_logging
from sp import sp_tickers
from utils import day_indicators

log = setup_logging("main.py")

# API_KEY = os.getenv("API_KEY")
# API_SECRET = os.getenv("API_SECRET")
API_KEY = "PK3D0P0EF5NVU7LKHY76"
API_SECRET = "X2kmdCqfYzGxaCYG2C3UhQ9DqHT9bYhYUhXM2g6G"
LR = 0.001  # Learning rate
HIDDEN_SIZE = 64
NUM_LAYERS = 2
NUM_EPOCHS = 100
TIME_PERIODS = {"DAY": 1, "WEEK": 5, "MONTH": 20, "YEAR": 252}

if os.path.exists("assets/data.pkl"):
    data = pickle.load(open("assets/data.pkl", "rb"))
    log.info("Loaded data from pickle file.")
else:
    stocks = sp_tickers
    # Take every stock from russell_3000.csv, each ticker has its own row with a single column
    # stocks = [
    #     ticker for ticker in pd.read_csv("russell_3000.csv", header=None)[0].values
    # ]

    log.info(f"stocks: {stocks}")

    client = StockHistoricalDataClient(
        API_KEY,
        API_SECRET,
        url_override="https://data.alpaca.markets",
    )

    # Retrieve 20 days (one month) of price data.
    start_date = f"2022-01-01 00:00:00"
    end_date = f"2022-01-21 00:00:00"

    params = StockBarsRequest(
        symbol_or_symbols=stocks,
        start=start_date,
        end=end_date,
        timeframe=TimeFrame.Day,
    )

    data = client.get_stock_bars(params).data

    pickle.dump(data, open("assets/data.pkl", "wb"))

# log.info(data)

# log.info(f"data: {data}")

# Data Preparation
# Load and preprocess your financial data, calculate technical indicators

# Calculate day trading indicators
if os.path.exists("assets/indicators.pkl"):
    data = pickle.load(open("assets/indicators.pkl", "rb"))
    log.info("Loaded indicators from pickle file.")
else:
    data = list(map(lambda key: day_indicators(data[key]), data.keys()))
    pickle.dump(data, open("assets/indicators.pkl", "wb"))

log.info(len(data))
log.info(list(len(stock[0]) for stock in data))

if os.path.exists("assets/input_data.pkl"):
    input_data = pickle.load(open("assets/input_data.pkl", "rb"))
    log.info("Loaded input_data from pickle file.")
else:
    input_data = list(
        map(lambda stock: torch.nan_to_num(torch.tensor(stock), nan=0.0), data)
    )

    max_length = max(stock.shape[1] for stock in input_data)

    # Pad all stocks to the maximum length
    padded_data = [
        torch.cat([stock, torch.zeros(len(stock), max_length - stock.size(1))], dim=1)
        for stock in input_data
    ]

    # Convert to a list of sequences (as Long Tensors)
    sequences = torch.cat([torch.LongTensor([stock.size(1)]) for stock in input_data])

    # Create a batch of padded sequences
    batch = pad_sequence(padded_data, batch_first=True)

    # Pack the padded batch
    packed_batch = pack_padded_sequence(
        batch, sequences, batch_first=True, enforce_sorted=False
    )

    input_data = packed_batch.data

    # Log the shape of the padded data
    log.info(f"padded_data: {input_data}, \npadded_data.shape: {input_data.shape}")

    pickle.dump(input_data, open("assets/input_data.pkl", "wb"))

log.info(f"input_data: {input_data}")
log.info(f"input_data.shape: {input_data.shape}")

log.info(stock.shape for stock in input_data)

# Feature Selection
# selected_features = [
#     "SMA5",
#     "SMA10",
#     "BB_upper",
#     "BB_middle",
#     "BB_lower",
#     "CCI",
#     "ROC5",
#     "ROC10",
#     "RSI5",
#     "RSI10",
#     "FastK",
#     "SlowK",
#     "MFI5",
#     "MFI10",
#     "SAR",
# ]
# input_data = data[selected_features].values

print(input_data.shape)

# Data Sequencing
sequence_length = TIME_PERIODS["WEEK"]  # Adjust based on your strategy
X, y = [], []
for i in range(len(input_data) - sequence_length):
    X.append(input_data[i : i + sequence_length])
    # print(
    #     input_data[i : i + sequence_length].shape,
    #     input_data[i + sequence_length].shape,
    #     "\n",
    #     input_data[i : i + sequence_length],
    #     input_data[i + sequence_length],
    # )
    y.append(input_data[i + sequence_length])

# Train-Test Split
split_ratio = 0.8
# split = int(split_ratio * len(X))

X_train, X_val, y_train, y_val = train_test_split(
    X, y, shuffle=False, train_size=split_ratio
)

# log.info(f"X_train: {X_train}")
log.info(f"X_test: {X_val}")
# log.info(f"y_train: {y_train}")
log.info(f"y_val: {y_val}")

# Convert data to PyTorch tensors and create DataLoader for batching
train_dataset = TensorDataset(torch.stack(X_train), torch.stack(y_train))
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

test_dataset = TensorDataset(torch.stack(X_val), torch.stack(y_val))
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# Define LSTM Model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Forward pass
        # x: Input data of shape (batch_size, sequence_length, input_size)
        # Output: Predictions of shape (batch_size, sequence_length, output_size)

        # Pass the input through the LSTM layer
        print(f"x.shape: {x.shape}, x: {x}")
        lstm_out, _ = self.lstm(x)

        print(f"lstm_out.shape: {lstm_out.shape}, lstm_out: {lstm_out}")

        # Pass the LSTM output through the fully connected layer (linear layer)
        output = self.fc(lstm_out[:, -1])  # Select the last time step's output
        # print(f"output.shape: {output.shape}, output: {output}")

        return output


# Loss Function and Optimizer
model = LSTM(
    input_size=13 * TIME_PERIODS["WEEK"],
    hidden_size=13,
    num_layers=2,
    output_size=13,
)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_losses = []

# Training Loop
epoch_losses = []

if os.path.exists("assets/train_losses.pkl"):
    train_losses = pickle.load(open("assets/train_losses.pkl", "rb"))
    log.info("Loaded train_losses from pickle file.")

if os.path.exists("assets/epoch_losses.pkl"):
    epoch_losses = pickle.load(open("assets/epoch_losses.pkl", "rb"))
    log.info("Loaded epoch_losses from pickle file.")

if os.path.exists("assets/model.pkl"):
    model.load_state_dict(torch.load("assets/model.pkl"))
    log.info("Loaded model from pickle file.")
else:
    log.info("Training model...")

    for epoch in tqdm(range(NUM_EPOCHS)):
        epoch_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.float())
            # log.info(f"outputs: {outputs.shape},\n labels: {labels.shape}")
            loss = criterion(outputs.view(-1, 1), labels.float())
            loss.backward()
            optimizer.step()

            # Append the loss value to the list
            train_losses.append(loss.item())
            epoch_loss += loss.item()

        epoch_losses.append(epoch_loss / len(train_loader))

# Save training losses, epoch losses, and model to pkl files
try:
    pickle.dump(train_losses, open("assets/train_losses.pkl", "wb"))
    pickle.dump(epoch_losses, open("assets/epoch_losses.pkl", "wb"))
    torch.save(model.state_dict(), "assets/model.pkl")
except Exception as e:
    log.error(f"Unable to save losses or model: {e}")

try:
    plt.plot(range(NUM_EPOCHS), epoch_losses, label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
except Exception as e:
    log.error(f"Unable to plot training loss: {e}")

# Make predictions on the test set
# test_predictions = []
#
# try:
#     with torch.no_grad():
#         for inputs, labels in test_loader:
#             outputs = model(inputs.float())
#             test_predictions.extend(outputs.float().view(-1).tolist())
# except Exception as e:
#     log.error(f"Unable to make predictions: {e}")

test_predictions = []
test_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs.float())
        test_predictions.extend(outputs.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

# Converting lists to numpy arrays for easier manipulation
test_predictions = np.array(test_predictions)
test_labels = np.array(test_labels)

print(test_predictions.shape, test_labels.shape)

# Calculating Regression Metrics
mse = mean_squared_error(test_labels, test_predictions)
mae = mean_absolute_error(test_labels, test_predictions)
r2 = r2_score(test_labels, test_predictions)

# Logging the metrics
log.info(f"Mean Squared Error (MSE): {mse:.2f}")
log.info(f"Mean Absolute Error (MAE): {mae:.2f}")
log.info(f"R^2 Score: {r2:.2f}")

# Create a confusion matrix heatmap
try:
    cm = confusion_matrix(y_val, test_predictions)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()
except Exception as e:
    log.error(f"Unable to plot confusion matrix: {e}")

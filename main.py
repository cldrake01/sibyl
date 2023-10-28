import os
import torch
import talib
import pickle
import numpy as np
import pandas as pd
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
)

from log import setup_logging
from sp import sp_tickers
from utils import day_indicators

log = setup_logging("main.py")

time_periods = {"DAY": 1, "WEEK": 5, "MONTH": 20, "YEAR": 252}

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
        "PK3D0P0EF5NVU7LKHY76",
        "X2kmdCqfYzGxaCYG2C3UhQ9DqHT9bYhYUhXM2g6G",
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

    data = client.get_stock_bars(params)

    pickle.dump(data, open("assets/data.pkl", "wb"))

# log.info(data)

data = data.data

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
        torch.cat([stock, torch.zeros(15, max_length - stock.size(1))], dim=1)
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

# Assuming input_data is of shape (num_stocks, 15, 1533)
# input_data = input_data.permute(1, 2, 0)

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

# Data Sequencing
sequence_length = 10  # Adjust based on your strategy
X, y = [], []
for i in range(len(input_data) - sequence_length):
    X.append(input_data[i: i + sequence_length])
    y.append(input_data[i + sequence_length])

# Train-Test Split
split_ratio = 0.8
split = int(split_ratio * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# log.info(f"X_train: {X_train}")
log.info(f"X_test: {X_test}")
# log.info(f"y_train: {y_train}")
log.info(f"y_test: {y_test}")

# Data Scaling
# Normalize or scale your data

# Convert data to PyTorch tensors and create DataLoader for batching
train_dataset = TensorDataset(torch.stack(X_train), torch.stack(y_train))
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

test_dataset = TensorDataset(torch.stack(X_test), torch.stack(y_test))
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
        lstm_out, _ = self.lstm(x)

        # Pass the LSTM output through the fully connected layer (linear layer)
        output = self.fc(lstm_out[:, -1, :])  # Select the last time step's output

        return output


# Loss Function and Optimizer
model = LSTM(
    input_size=time_periods["MONTH"],
    hidden_size=64,
    num_layers=2,
    output_size=time_periods["WEEK"],
)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_losses = []

# Training Loop
num_epochs = 1_000  # Adjust based on your strategy
epoch_losses = []

for epoch in tqdm(range(num_epochs)):
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

plt.plot(range(num_epochs), train_losses, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Make predictions on the test set
test_predictions = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs.float())
        predictions = (outputs > 0.8).float()  # Adjust threshold as needed
        test_predictions.extend(predictions.view(-1).tolist())

# Calculate performance metrics
accuracy = accuracy_score(y_test, test_predictions)
precision = precision_score(y_test, test_predictions)
recall = recall_score(y_test, test_predictions)
f1 = f1_score(y_test, test_predictions)

log.info(f"Accuracy: {accuracy:.2f}")
log.info(f"Precision: {precision:.2f}")
log.info(f"Recall: {recall:.2f}")
log.info(f"F1-Score: {f1:.2f}")

# Create a confusion matrix heatmap
cm = confusion_matrix(y_test, test_predictions)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

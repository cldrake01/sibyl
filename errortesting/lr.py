import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def vmse(y_true, y_pred):
    variance_y = torch.var(y_true)
    variance_y_hat = torch.var(y_pred)
    max_abs_error = torch.max(torch.abs(y_true - y_pred))
    return torch.exp(torch.abs(variance_y - variance_y_hat)) * max_abs_error


class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


def train_model(model, X_train, y_train, learning_rate=0.01, num_epochs=100):
    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    # Define optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        # Forward pass
        predictions = model(X_train_tensor)
        loss = vmse(y_train_tensor, predictions)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model

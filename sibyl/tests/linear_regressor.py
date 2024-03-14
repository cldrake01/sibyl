import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

from sibyl.utils.loss import MaxSE


class LinearRegressor(nn.Module):
    def __init__(self):
        super(LinearRegressor, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


def train_model(
    model: nn.Module,
    X: Tensor,
    y: Tensor,
    learning_rate: float = 0.01,
    num_epochs: int = 100,
):
    loss_function: callable = MaxSE()

    # Define optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        # Forward pass
        y_hat = model(X)
        loss = loss_function(y, y_hat)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def main():
    np.random.seed(0)
    X = torch.tensor(np.random.rand(100, 1))
    y = torch.tensor(3 * X + 2 + np.random.randn(100, 1) * 0.1)

    # Train the model
    model = train_model(LinearRegressor(), X, y)

    # Print the model parameters
    print(list(model.parameters()))


if __name__ == "__main__":
    main()

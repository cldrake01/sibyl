import torch
import torch.nn as nn


class LinearRegressor(nn.Module):
    def __init__(self, in_dims: int, out_dims, *, layers: int = 1, dropout: float = 0):
        super(LinearRegressor, self).__init__()
        self.linear = nn.Linear(in_dims, out_dims)

    def forward(self, x):
        x = self.linear(x.mT)
        return x.mT

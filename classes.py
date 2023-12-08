import torch

from torch import nn
from dataclasses import dataclass
from torch.utils.data import Dataset


class Vec(nn.Module):
    def __init__(self, input_size):
        super(Vec, self).__init__()
        self.lstm1 = nn.LSTM(input_size, 32, batch_first=True)  # Reduced complexity
        self.dropout1 = nn.Dropout(0.5)  # Added dropout
        self.lstm2 = nn.LSTM(32, 32, batch_first=True)  # Reduced complexity
        self.dropout2 = nn.Dropout(0.5)  # Added dropout
        self.linear = nn.Linear(32, 15)

    def forward(self, input, future=0):
        outputs = []
        h_t, _ = self.lstm1(input)
        h_t = self.dropout1(h_t)
        h_t2, _ = self.lstm2(h_t)
        h_t2 = self.dropout2(h_t2)
        output = self.linear(h_t2[:, -1, :])
        outputs += [output]
        for i in range(future):  # if we should predict the future
            h_t, _ = self.lstm1(output.unsqueeze(1))
            h_t2, _ = self.lstm2(h_t)
            output = self.linear(h_t2[:, -1, :])
            outputs += [output]
        outputs = torch.cat(outputs, dim=1)
        return outputs


class MetaLabeler(nn.Module):
    """
    A transformer model for determining the magnitude of Vec's and RVec's proposed trades. MetaLabeler is trained
    upon Vec's and RVec's predictions.
    """

    def __init__(self, *args, **kwargs):
        super(MetaLabeler, self).__init__()


class TimeSeriesDataset(Dataset):
    def __init__(self, data, input_window_size=60, output_window_size=15):
        self.data = data
        self.input_window_size = input_window_size
        self.output_window_size = output_window_size
        self.total_window_size = input_window_size + output_window_size
        self.num_stocks = data.shape[0]
        self.num_indicators = data.shape[1]
        self.num_time_steps = data.shape[2]

    def __len__(self):
        num_windows_per_stock = self.num_time_steps - self.total_window_size + 1
        return self.num_stocks * max(num_windows_per_stock, 0)

    def __getitem__(self, index):
        stock_index = index // (self.num_time_steps - self.total_window_size + 1)
        time_index = index % (self.num_time_steps - self.total_window_size + 1)

        input_window = self.data[
            stock_index, :, time_index : time_index + self.input_window_size
        ]
        output_window = self.data[
            stock_index,
            :,
            time_index + self.input_window_size : time_index + self.total_window_size,
        ]

        return input_window, output_window


@dataclass
class Bar:
    """
    A bar is a single unit of time series data.
    """

    open: float
    high: float
    low: float
    close: float
    volume: int

    def __dict__(self):
        return {
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


class NullLogger:
    def __init__(self, *args, **kwargs):
        ...

    def info(self, *args, **kwargs):
        ...

    def debug(self, *args, **kwargs):
        ...

    def warning(self, *args, **kwargs):
        ...

    def error(self, *args, **kwargs):
        ...

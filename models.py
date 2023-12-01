import torch
from torch import nn


class Vec(nn.Module):
    """
    A transformer model for predicting an asset's directionality (up or down) in the next time step.

    Vec is trained upon smoothed data, as opposed to RVec's raw data.
    """

    def __init__(self):
        super(Vec, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

    def forward(self, input, future=0):
        outputs = []
        h_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        h_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)

        for input_t in input.split(1, dim=1):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        for i in range(future):  # if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.cat(outputs, dim=1)
        return outputs


class RVec(Vec):
    """
    A transformer model for predicting black swan events.

    RVec is trained upon raw data, as opposed to Vec's smoothed data.
    """

    def __init__(self):
        super(RVec, self).__init__()


class MetaLabeler(nn.Module):
    """
    A transformer model for determining the magnitude of Vec's and RVec's proposed trades. MetaLabeler is trained
    upon Vec's and RVec's predictions.
    """

    def __init__(self, *args, **kwargs):
        super(MetaLabeler, self).__init__()

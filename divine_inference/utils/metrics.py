import numpy as np


def rse(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(
        np.sum((true - true.mean()) ** 2)
    )


def corr(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def mae(pred, true):
    return np.mean(np.abs(pred - true))


def mse(pred, true):
    return np.mean((pred - true) ** 2)


def rmse(pred, true):
    return np.sqrt(mse(pred, true))


def mape(pred, true):
    return np.mean(np.abs((pred - true) / true))


def mspe(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    return (
        rse(pred, true),
        corr(pred, true),
        mae(pred, true),
        mse(pred, true),
        rmse(pred, true),
        mape(pred, true),
        mspe(pred, true),
    )

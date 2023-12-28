import json
from collections import deque

import torch

from sibyl import logger
from sibyl.utils import Bar
from sibyl.utils.preprocessing import stock_tensors
from sibyl.utils.tickers import tickers
from sibyl.utils.weights import download_weights

API_KEY = "CK3D0VVO5962GGQMX47C"
API_SECRET = "1ywZ2YMpNdGklgKvJ3heyGsisMVOYWqDvFyGgCXC"

log = logger("inference.py")

WINDOW_SIZE = 60
# TA-Lib represents all values prior to the calculation interval value as NaN (they're zeroed out).
# Therefore, we need to add the greatest interval to the window size.
# Otherwise, we'll get a tensor whose first 14 values are zeros, like so:
# tensor([[[0., 0., 0.,  ..., 0.22, 2, 3],
#          [0., 0., 0.,  ..., 4, 7, 2],
#          [0., 0., 0.,  ..., 5, 2, 1],
#          ...,
#          [0., 0., 0.,  ..., 2, 1, 1],
#          [0., 0., 0.,  ..., 1, 2, 0],
#          [0., 0., 0.,  ..., 2, 1, 1]]])
GREATEST_INTERVAL = 14  # See indicators()

rolling_windows = {ticker: deque(maxlen=WINDOW_SIZE + GREATEST_INTERVAL) for ticker in tickers}


# Retrieve model weights from GCS
log.info("Retrieving model weights from GCS...")
weights = download_weights("gs://informer-weights/weights.pt")

# Load the model weights
log.info("Loading model weights...")
model = weights.load_state_dict(torch.load("informer.pt"))
log.info("Model weights loaded.")

# Convert the model to eval mode
model.eval()


def bar(stock: dict) -> Bar:
    return Bar(
        open=stock["o"],
        close=stock["c"],
        high=stock["h"],
        low=stock["l"],
        volume=stock["v"],
    )


def on_open(ws):
    log.info("Opened connection")

    # Authentication and subscribing to a channel
    auth_data = {"action": "auth", "key": f"{API_KEY}", "secret": f"{API_SECRET}"}
    ws.send(json.dumps(auth_data))

    # Subscribe to some market data
    listen_message = {
        "action": "subscribe",
        "bars": tickers,
    }
    ws.send(json.dumps(listen_message))


def on_message(ws, message):
    # Add latest bar to the rolling window
    for stock in latest := json.loads(message):
        rolling_windows[stock["S"]].append(bar(stock))

    # Predict the next 15 bars
    for stock in latest:
        if len(rolling_windows[stock["S"]]) == 60:
            # Convert the rolling window to a tensor
            # NOTE: stock_tensors() returns a rank-three tensor, which is essentially an entire dataset.
            #       We only want the last tensor in the dataset, which is the tensor for the rolling window.
            #       Therefore, we use [-1] to get the last tensor in the dataset.
            X, _ = stock_tensors([rolling_windows[stock["S"]]])[-1]  # (1, window_size, indicators)
            # Predict the next 15 bars
            y_hat = model(X)


def on_error(ws, error):
    log.error(error)


def on_close(ws):
    log.warning("Closed connection")


if __name__ == "__main__":
    ...
    # ws_ = websocket.WebSocketApp(
    #     "wss://stream.data.sandbox.alpaca.markets/v2/iex",
    #     on_open=on_open,
    #     on_message=on_message,
    #     on_error=on_error,
    #     on_close=on_close,
    # )
    # ws_.run_forever()
    #
    # test_data = [
    #     {
    #         "T": "b",
    #         "S": "MSFT",
    #         "o": 371.47,
    #         "h": 371.47,
    #         "l": 371.47,
    #         "c": 371.47,
    #         "v": 103,
    #         "t": "2023-12-05T19:43:00Z",
    #         "n": 4,
    #         "vw": 371.471602,
    #     },
    #     {
    #         "T": "b",
    #         "S": "VNT",
    #         "o": 33.95,
    #         "h": 33.95,
    #         "l": 33.95,
    #         "c": 33.95,
    #         "v": 200,
    #         "t": "2023-12-05T19:43:00Z",
    #         "n": 2,
    #         "vw": 33.95,
    #     },
    #     {
    #         "T": "b",
    #         "S": "EIX",
    #         "o": 66.78,
    #         "h": 66.78,
    #         "l": 66.78,
    #         "c": 66.78,
    #         "v": 111,
    #         "t": "2023-12-05T19:43:00Z",
    #         "n": 3,
    #         "vw": 66.780495,
    #     },
    #     {
    #         "T": "b",
    #         "S": "ZBH",
    #         "o": 115.82,
    #         "h": 115.82,
    #         "l": 115.81,
    #         "c": 115.81,
    #         "v": 500,
    #         "t": "2023-12-05T19:43:00Z",
    #         "n": 14,
    #         "vw": 115.8126,
    #     },
    #     {
    #         "T": "b",
    #         "S": "MRNA",
    #         "o": 78.42,
    #         "h": 78.42,
    #         "l": 78.42,
    #         "c": 78.42,
    #         "v": 161,
    #         "t": "2023-12-05T19:43:00Z",
    #         "n": 5,
    #         "vw": 78.381242,
    #     },
    #     {
    #         "T": "b",
    #         "S": "C",
    #         "o": 46.825,
    #         "h": 46.825,
    #         "l": 46.825,
    #         "c": 46.825,
    #         "v": 154,
    #         "t": "2023-12-05T19:43:00Z",
    #         "n": 2,
    #         "vw": 46.826753,
    #     },
    #     {
    #         "T": "b",
    #         "S": "MS",
    #         "o": 80.23,
    #         "h": 80.23,
    #         "l": 80.215,
    #         "c": 80.215,
    #         "v": 372,
    #         "t": "2023-12-05T19:43:00Z",
    #         "n": 5,
    #         "vw": 80.219987,
    #     },
    #     {
    #         "T": "b",
    #         "S": "HST",
    #         "o": 17.65,
    #         "h": 17.65,
    #         "l": 17.65,
    #         "c": 17.65,
    #         "v": 110,
    #         "t": "2023-12-05T19:43:00Z",
    #         "n": 2,
    #         "vw": 17.649545,
    #     },
    #     {
    #         "T": "b",
    #         "S": "VRSN",
    #         "o": 215.925,
    #         "h": 215.925,
    #         "l": 215.925,
    #         "c": 215.925,
    #         "v": 268,
    #         "t": "2023-12-05T19:43:00Z",
    #         "n": 3,
    #         "vw": 215.925,
    #     },
    #     {
    #         "T": "b",
    #         "S": "MRO",
    #         "o": 24.64,
    #         "h": 24.64,
    #         "l": 24.63,
    #         "c": 24.63,
    #         "v": 1393,
    #         "t": "2023-12-05T19:43:00Z",
    #         "n": 20,
    #         "vw": 24.638564,
    #     },
    # ]
    #
    # latest = test_data
    #
    # # map
    # start = time.time()
    # _ = tuple(map(lambda stock: rolling_windows[stock["S"]].append(bar(stock)), latest))
    # end = time.time()
    # log.info(f"Map: {end - start}")
    #
    # # generator
    # start = time.time()
    # _ = tuple(rolling_windows[stock["S"]].append(bar(stock)) for stock in latest)
    # end = time.time()
    # log.info(f"Generator: {end - start}")
    #
    # # for loop
    # start = time.time()
    # for stock in latest:
    #     rolling_windows[stock["S"]].append(bar(stock))
    # end = time.time()
    # log.info(f"For loop: {end - start}")
    #
    # pprint(rolling_windows)

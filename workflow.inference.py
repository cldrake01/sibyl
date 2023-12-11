import json

import websocket

from log import *
from tickers import tickers
from utils import *

API_KEY = "CK3D0VVO5962GGQMX47C"
API_SECRET = "1ywZ2YMpNdGklgKvJ3heyGsisMVOYWqDvFyGgCXC"

log: logging.Logger = setup_logging("test.inference.py")


def on_open(ws):
    log.info("Opened connection")

    # Authentication and subscribing to a channel
    auth_data = {"action": "auth", "key": f"{API_KEY}", "secret": f"{API_SECRET}"}
    ws.send(json.dumps(auth_data))

    # Retrieve model weights from GCS
    log.info("Retrieving model weights from GCS...")
    model = download_weights("gs://informer-weights/weights.pt")

    # Subscribe to some market data
    listen_message = {
        "action": "subscribe",
        "bars": tickers,
    }
    ws.send(json.dumps(listen_message))


def on_message(ws, message):
    data = json.loads(message)
    stock_bars = [
        {
            stock["S"]: [
                Bar(
                    open=stock["o"],
                    close=stock["c"],
                    high=stock["h"],
                    low=stock["l"],
                    volume=stock["v"],
                )
            ]
            for stock in data
        }
    ]


def on_error(ws, error):
    log.error(error)


def on_close(ws):
    log.warning("Closed connection")

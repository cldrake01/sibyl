import os.path

from dotenv import load_dotenv

from sibyl.utils.loss import VMaxAE
from sibyl.utils.models.informer.model import Informer
from sibyl.utils.tickers import tickers

load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_WEBSOCKET_KEY = os.getenv("ALPACA_WEBSOCKET_KEY")
ALPACA_WEBSOCKET_SECRET = os.getenv("ALPACA_WEBSOCKET_SECRET")

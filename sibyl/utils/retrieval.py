from datetime import datetime, timedelta

from alpaca.data import TimeFrame, StockBarsRequest, StockHistoricalDataClient
from tqdm import tqdm

from sibyl import ALPACA_API_KEY, ALPACA_API_SECRET
from sibyl.utils.config import Config
from sibyl.utils.tickers import tickers


def alpaca_time_series(
    stocks: list[str], start: datetime | str, end: datetime | str
) -> dict:
    """
    Retrieve time series data from the Alpaca API.

    :param stocks: (list[str]): A list of stock tickers.
    :param start: (datetime or str): Start date.
    :param end: (datetime or str): End date.
    :return: dict: A dictionary of stock data.
    """

    # Alpaca's API, whilst it does accept datetime objects, is prone to error when using them.
    if isinstance(start, datetime):
        start = start.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(end, datetime):
        end = end.strftime("%Y-%m-%d %H:%M:%S")

    client = StockHistoricalDataClient(
        api_key=ALPACA_API_KEY,
        secret_key=ALPACA_API_SECRET,
    )

    # url_override="https://data.alpaca.markets",

    params = StockBarsRequest(
        symbol_or_symbols=stocks,
        start=start,
        end=end,
        timeframe=TimeFrame.Minute,
    )

    return client.get_stock_bars(params).data


def fetch_data(config: Config) -> list:
    """
    Retrieve data from the Alpaca API for a given number of years using multiple worker-threads.

    :param config: (TrainingConfig): A configuration object for time series data.
    :return: list: A list of bars for a given stock.
    """
    config.log.info("Retrieving data from the Alpaca API...")

    # Partitioning the tickers
    partitions = [
        tickers[i : i + config.rate] for i in range(0, len(tickers), config.rate)
    ]
    config.log.info(f"Partitioned tickers into {len(partitions)} partitions.")

    data = []

    for ticker in tqdm(tickers, desc="Retrieving Data"):
        data.append(
            alpaca_time_series(
                [ticker],
                datetime.today() - timedelta(365 * config.years),
                datetime.today(),
            )
        )

    config.log.info("Retrieved data from the Alpaca API.")

    return data

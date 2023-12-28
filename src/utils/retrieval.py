from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

from alpaca.data import TimeFrame, StockBarsRequest, StockHistoricalDataClient

from src import NullLogger
from src.utils import API_KEY, API_SECRET
from src.utils.tickers import tickers


def alpaca_time_series(
    stocks: list[str], start: datetime or str, end: datetime or str
) -> dict:
    """
    Retrieve time series data from the Alpaca API.

    :param stocks: (list[str]): A list of stock tickers.
    :param start: (datetime or str): Start date.
    :param end: (datetime or str): End date.
    :return: dict: A dictionary of stock data.
    """

    if isinstance(start, datetime):
        start = start.strftime("%Y-%m-%d %H:%M:%S")

    if isinstance(end, datetime):
        end = end.strftime("%Y-%m-%d %H:%M:%S")

    client = StockHistoricalDataClient(
        API_KEY,
        API_SECRET,
    )

    # url_override="https://data.alpaca.markets",

    params = StockBarsRequest(
        symbol_or_symbols=stocks,
        start=start,
        end=end,
        timeframe=TimeFrame.Minute,
    )

    return client.get_stock_bars(params).data


def fetch_data(
    years: int or float,
    max_workers: int = len(tickers) // 2,
    log=NullLogger(),
) -> list:
    """
    Retrieve data from the Alpaca API for a given number of years using multiple worker-threads.

    :param years: (float): Number of years of data to retrieve.
    :param max_workers: (int): Maximum number of worker-threads to use.
    :param log: (logging.Logger): Logger (optional).
    :return: list: A list of bars for a given stock.
    """
    # Retrieve data from the Alpaca API
    log.info("Retrieving data from the Alpaca API...")
    log.info(f"Maximum Worker-threads: {max_workers}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                alpaca_time_series,
                [ticker],
                datetime.today() - timedelta(365 * years),
                datetime.today(),
            )
            for ticker in tickers
        ]

    # Collecting results
    data = [future.result() for future in futures]
    log.info("Retrieved data from the Alpaca API.")

    return data

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

from alpaca.data import TimeFrame, StockBarsRequest, StockHistoricalDataClient

from sibyl import ALPACA_API_KEY, ALPACA_API_SECRET, TimeSeriesConfig
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
        ALPACA_API_KEY,
        ALPACA_API_SECRET,
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
    config: TimeSeriesConfig,
) -> list:
    """
    Retrieve data from the Alpaca API for a given number of years using multiple worker-threads.

    :param config: (TimeSeriesConfig): A configuration object for time series data.
    :return: list: A list of bars for a given stock.
    """
    # Retrieve data from the Alpaca API
    config.log.info("Retrieving data from the Alpaca API...")
    config.log.info(f"Maximum Worker-threads: {config.max_workers}")

    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        futures = [
            executor.submit(
                alpaca_time_series,
                [ticker],
                datetime.today() - timedelta(365 * config.years),
                datetime.today(),
            )
            for ticker in tickers
        ]

    # Collecting results
    data = [future.result() for future in futures]
    config.log.info("Retrieved data from the Alpaca API.")

    return data

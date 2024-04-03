from datetime import datetime, timedelta

from alpaca.data import TimeFrame, StockBarsRequest, StockHistoricalDataClient
from tqdm import tqdm

from sibyl import ALPACA_API_KEY, ALPACA_API_SECRET
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


def fetch_data(config: "Config") -> list:
    """
    Retrieve data from the Alpaca API for a given number of years using multiple worker-threads.

    :param config: (TrainingConfig): A configuration object for time series data.
    :return: list: A list of bars for a given stock.
    """
    config.log.info("Retrieving data from the Alpaca API...")

    # On weekends, we must adjust the date range to avoid errors
    match datetime.today().weekday():
        case 5:
            end = datetime.today() - timedelta(1)
            config.log.warning("It's Saturday. Adjusting the date range...")
        case 6:
            end = datetime.today() - timedelta(2)
            config.log.warning("It's Sunday. Adjusting the date range...")
        case _:
            end = datetime.today()

    data: list[dict] = [
        alpaca_time_series([ticker], end - timedelta(365 * config.years), end)
        for ticker in tqdm(tickers, desc="Fetching data...")
    ]

    unfiltered_len = len(data)

    data = list(filter(lambda x: len(x.values()) < config.feature_window_size, data))

    if unfiltered_len - len(data) > 0:
        config.log.warning(f"Filtered out {unfiltered_len - len(data)} results.")

    config.log.info("Retrieved data from the Alpaca API.")

    return data

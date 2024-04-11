from dataclasses import dataclass


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

    @staticmethod
    def from_dict(data: dict):
        return Bar(
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            volume=data["volume"],
        )

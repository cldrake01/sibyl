from log import *
from utils import *

log: logging.Logger = setup_logging("main.py")

time_series = fetch_data(years=0.1, log=log)

train(time_series=time_series, epochs=2_000, log=log)

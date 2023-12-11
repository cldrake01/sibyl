import os
import pickle

from classes import Informer
from log import setup_logging
from utils import fetch_data, stock_tensors, train
from transformers import (
    InformerForPrediction,
    InformerConfig,
)

log = setup_logging("workflow.train.py")

log.info("Checking for pickle file...")

path = "assets/pkl/time_series.pkl"

if not os.path.exists(path):
    log.info("Pickle file not found. Fetching data...")
    time_series = fetch_data(years=0.1, log=log)
    log.info("Creating pickle file...")
    with open(path, "wb") as f:
        pickle.dump(time_series, f)
else:
    log.info("Pickle file found.")
    log.info("Loading pickle file...")
    with open(path, "rb") as f:
        time_series = pickle.load(f)

log.info("Creating tensors...")
X, y = stock_tensors(time_series)
log.info("Tensors created.")

model = Informer(
    input_size=X.shape[-1],  # 8
    output_size=y.shape[-1],  # 8
    d_model=512,
    n_heads=8,
    e_layers=2,
    d_layers=1,
    dropout=0.1,
)

# Train the model
log.info("Starting training...")
train(X=X, y=y, model=model, epochs=2_000, log=log)

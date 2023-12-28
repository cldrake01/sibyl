import torch
from google.cloud import storage

from sibyl import NullLogger
from sibyl.utils.models.model import Informer


def upload_weights(bucket_name: str, filename: str = "informer.pt"):
    """
    Upload model weights to GCS.

    :param bucket_name: (str): Name of the GCS bucket.
    :param filename: (str): Name of the file to upload.
    """
    # Instantiates a client
    client = storage.Client()

    bucket = client.get_bucket(bucket_name)

    # Upload the latest model weights
    blob = bucket.blob(filename)
    blob.upload_from_filename(filename)


def download_weights(
    bucket_name: str, filename: str = "informer.pt", log=NullLogger()
) -> Informer:
    """
    Retrieve model weights from GCS.

    :param bucket_name: (str): Name of the GCS bucket.
    :param filename: (str): Name of the file to download.
    :param log: (logging.Logger): Logger (optional).
    :return: Informer: A PyTorch model.
    """
    # Instantiates a client
    client = storage.Client()

    try:
        bucket = client.get_bucket(bucket_name)
    except Exception as e:
        log.error(f"Unable to retrieve bucket from GCS with {bucket_name}.")
        log.error(e)
        raise e

    # Download the latest model weights
    try:
        blobs = bucket.list_blobs()
        latest_blob = max(blobs, key=lambda blob: blob.time_created)
        latest_blob.download_to_filename(filename)
    except Exception as e:
        log.error(f"Unable to download model weights from GCS.")
        log.error(e)
        raise e

    # Load the model weights
    try:
        model = Informer(
            input_size=8,
            output_size=8,
            d_model=512,
            n_heads=8,
            e_layers=2,
            d_layers=1,
            dropout=0.1,
        )
        model.load_state_dict(torch.load(filename))

        return model
    except Exception as e:
        log.error(f"Unable to load model weights from {filename}.")
        log.error(e)
        raise e

import logging


def setup_logging(file_name: str) -> logging.Logger:
    """
    Setup logging configuration

    :param file_name: Path to the log file
    """
    file_name = f"logs/{file_name}.log"

    # Create a logger
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.DEBUG)

    # Create a file handler to log to a text file
    file_handler = logging.FileHandler(file_name)

    # Create a formatter to format log messages
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Set the formatter for the file handler
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    # Send an initial message
    logger.info(f"Logging to {file_name}")

    return logger

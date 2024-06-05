import logging
import os
from logging.handlers import TimedRotatingFileHandler
import datetime

def configure_logging(log_directory):
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Create console handler and set level to INFO
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create file handler and set level to INFO
    os.makedirs(log_directory, exist_ok=True)
    log_file = os.path.join(log_directory, datetime.datetime.now().strftime("%d-%m-%Y") + ".log")
    file_handler = TimedRotatingFileHandler(log_file, when="midnight", interval=1, backupCount=7)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Add a handler for errors
    error_file = os.path.join(log_directory, "errors.log")
    error_handler = TimedRotatingFileHandler(error_file, when="midnight", interval=1, backupCount=7)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)
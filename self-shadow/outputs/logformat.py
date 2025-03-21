import logging

class CustomFormatter(logging.Formatter):
    def __init__(self_formatter):
        pass

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = '%(asctime)s | %(levelname)s | %(message)s'

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self_formatter, record):
        log_fmt = self_formatter.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt,datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)

import logging


def setup_logger(log_file, log_level):
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level))
    handler = logging.FileHandler(log_file)
    handler.setLevel(getattr(logging, log_level))
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

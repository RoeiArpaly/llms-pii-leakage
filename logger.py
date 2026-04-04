"""Application-wide logger writing to main.log (file only).

Console output is handled by cli.py to avoid interleaving with
tqdm progress bars and the CLI display.
"""
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler("main.log")
formatter = logging.Formatter("[%(levelname)s][%(asctime)s]: %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

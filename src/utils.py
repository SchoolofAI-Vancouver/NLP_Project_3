#! /usr/bin/env python

from pathlib import Path
import logging


def get_root():
    """Return project root folder"""
    return Path(__file__).parent.parent


def get_logger(level=logging.INFO):
    """A simple logger for tracking training process"""
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info("Logging successfully configured!")

    return logger

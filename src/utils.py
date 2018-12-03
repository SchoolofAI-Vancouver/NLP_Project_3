#! /usr/bin/env python

from pathlib import Path


def get_root():
    """Return project root folder"""
    return Path(__file__).parent.parent

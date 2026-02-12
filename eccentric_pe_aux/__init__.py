"""
Eccentric PE Auxiliary Tools

Scripts and utilities for eccentric parameter estimation.
"""

__version__ = "0.1.0"

from . import savage_dickey_bf
from . import convert_posterior_egw
from . import convert_posterior_EOB

__all__ = [
    "savage_dickey_bf",
    "convert_posterior_egw",
    "convert_posterior_EOB",
]

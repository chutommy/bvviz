"""Provides helper functions."""

import numpy as np


def str_to_byte(string: str) -> np.array:
    """Encode string to numpy array of bytes."""
    return np.array(list(string), dtype=np.byte)


def byte_to_str(byte_code: np.array) -> str:
    """Decode numpy array of bytes to string."""
    result = ""
    for bit in byte_code:
        if bit == 0:
            result += "0"
        else:
            result += "1"

    return result

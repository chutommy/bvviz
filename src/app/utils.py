"""Provides utility and helper functions."""

from datetime import datetime
from random import randint
from typing import List

import numpy as np
from qiskit.providers import Backend

from .config import OptimizationLevel


def generate_seed() -> int:
    """Generates random seed."""
    return randint(10 ** 14, 10 ** 15)


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


def timestamp_str(time: datetime = datetime.now()) -> str:
    """Return string formatted time (filename-friendly)."""
    return time.strftime("%Y_%m_%d_%H-%M-%S")


def backend_to_name(backend: Backend) -> str:
    """Extract the name and number of qubits from the provider's fake backend system identifier."""
    # fake_######_v2
    name = backend.name
    if name.startswith("fake_"):
        name = name[5:]
    if name.endswith("_v2"):
        name = name[:-3]
        name = name.replace("_", " ")
    return f"{name.capitalize()} ({backend.num_qubits})"


def method_to_name(method: str) -> str:
    """Returns a formatted name of the method."""
    # fake_######_v2
    return method.replace("_", " ").capitalize()


def optimization_to_name(level: int) -> str:
    """Map the enum title to the value."""
    return OptimizationLevel(level).name.replace("_", " ").capitalize()


def all_measurement_outputs(complexity: int) -> List[str]:
    """Returns all possible outcomes of measurement of a system of size N."""
    outs = []
    tmp_str = list("0" * complexity)

    def update_pos_rec(strs: List[str], pos: int):
        if pos >= complexity:
            outs.append("".join(tmp_str))
            return

        update_pos_rec(strs, pos + 1)

        tmp_str[pos] = '1'
        update_pos_rec(strs, pos + 1)
        tmp_str[pos] = '0'

    update_pos_rec(tmp_str, 0)
    return outs


def fill_counts(counts: dict, size: int):
    """Fills counts with non-measured values."""
    all_meas = all_measurement_outputs(size)
    for meas in all_meas:
        if meas not in counts:
            counts[meas] = 0


def sort_zipped(xvalues: np.array, yvalues: np.array) -> (np.array, np.array):
    """Sort two lists based on one of them (xs)."""
    order = xvalues.argsort()
    return xvalues[order], yvalues[order]


def diff_letters(str1: str, str2: str) -> int:
    """Return number of different letters in strings a and b."""
    return sum(str1[i] != str2[i] for i in range(len(str1)))


def check_secret(secret: str) -> bool:
    """Verifies that the secret consists of only 0s and 1s."""
    return not all(c in '01' for c in secret)


def pct_to_str(pct, total):
    """Formats percentage to string."""
    absolute = int(np.round(pct * total / 100))
    return f"{pct:.2f}%\n({absolute:d} shots)"


def find_secret(arr: np.array, secret) -> int:
    """Finds position of the secret in the array."""
    pos = 0
    for i, val in enumerate(arr):
        if secret == val:
            pos = i
            break
    return pos


def dhash(dictionary: dict):
    """Computes hash for directory."""
    return hash(frozenset(dictionary.items()))
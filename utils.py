"""Provides helper functions."""
from datetime import datetime
from random import randint

import numpy as np
from qiskit.providers import Backend

from config import OptimizationLevel


def generate_seed() -> int:
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

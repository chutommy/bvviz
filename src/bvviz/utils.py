"""Provides utility and helper functions."""
from datetime import datetime
from random import randint
from typing import Any, Dict, List, Tuple

import numpy as np
import numpy.typing as npt
from qiskit.providers import Backend

from .config import OptimizationLevel


def generate_seed() -> int:
    """Generates random seed."""
    return randint(10 ** 14, 10 ** 15)


def str_to_byte(string: str) -> npt.NDArray[np.byte]:
    """Encode string to numpy array of bytes."""
    return np.array(list(string), dtype=np.byte)


def byte_to_str(byte_code: npt.NDArray[np.byte]) -> str:
    """Decode numpy array of bytes to string."""
    result = ''
    for bit in byte_code:
        if bit == 0:
            result += '0'
        else:
            result += '1'

    return result


def timestamp_str(time: datetime = datetime.now()) -> str:
    """Return string formatted time (filename-friendly)."""
    return time.strftime('%Y_%m_%d_%H-%M-%S')


def backend_name(backend: Backend) -> str:
    """Extract the name and number of qubits from the provider's fake backend system identifier."""
    # common format: fake_######_v2
    name = backend.name
    if name.startswith('fake_'):
        name = name[5:]
    if name.endswith('_v2'):
        name = name[:-3]
    name = name.replace('_', ' ')
    return f'{name.title()} ({backend.num_qubits})'


def method_name(method: str) -> str:
    """Returns a formatted name of the method."""
    return method.replace('_', ' ').title()


def optimization_name(level: int) -> str:
    """Map the enum title to the value."""
    return OptimizationLevel(level).name.replace('_', ' ').title()


def all_measurement_outputs(complexity: int) -> List[str]:
    """Returns all possible outcomes of measurement of a system of size N."""
    outs = []
    tmp_str = list('0' * complexity)

    def update_pos_rec(strs: List[str], pos: int) -> None:
        if pos >= complexity:
            outs.append(''.join(tmp_str))
            return

        update_pos_rec(strs, pos + 1)

        tmp_str[pos] = '1'
        update_pos_rec(strs, pos + 1)
        tmp_str[pos] = '0'

    update_pos_rec(tmp_str, 0)
    return outs


def fill_counts(counts: Dict[str, int], size: int) -> None:
    """Fills counts with non-measured values."""
    all_meas = all_measurement_outputs(size)
    for meas in all_meas:
        if meas not in counts:
            counts[meas] = 0


def sort_zipped(x_values: npt.NDArray[Any], y_values: npt.NDArray[Any]) \
        -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Sort two lists based on one of them (xs)."""
    order = x_values.argsort()
    return x_values[order], y_values[order]


def diff_letters(str1: str, str2: str) -> int:
    """Return number of different letters in strings a and b."""
    return sum(str1[i] != str2[i] for i in range(len(str1)))


def check_secret(secret: str) -> bool:
    """Verifies that the secret consists of only 0s and 1s."""
    return all(c in '01' for c in secret)


def pct_to_str(pct: float, total: int) -> str:
    """Formats percentage to string."""
    absolute = int(np.round(pct * total / 100))
    return f'{pct:.2f}%\n({absolute:d} shots)'


def find_secret(arr: npt.NDArray[Any], secret: str) -> int:
    """Finds position of the secret in the array."""
    for i, val in enumerate(arr):
        if secret == val:
            return i
    return len(arr)


def dict_max_value_key(dictionary: Dict[Any, Any]) -> Any:
    """Returns the key of the max value in the dictionary."""
    solution = ''
    max_count = 0
    for key, val in dictionary.items():
        if val > max_count:
            max_count = val
            solution = key
    return solution

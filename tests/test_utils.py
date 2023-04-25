import random
from datetime import datetime
from typing import Dict

import numpy as np
import numpy.typing as npt
import pytest
from qiskit.providers import Backend

from src.bvviz.utils import generate_seed, str_to_byte, byte_to_str, timestamp_str, backend_name, method_name, \
    all_measurement_outputs, fill_counts, sort_zipped, diff_letters, check_secret, find_secret, dict_max_value_key


def test_generate_seed():
    random.seed(42)
    for _ in range(10):
        assert generate_seed() != generate_seed()


@pytest.mark.parametrize('string, bytes', [
    ('0', np.array([0], dtype=np.byte)),
    ('1', np.array([1], dtype=np.byte)),
    ('1010', np.array([1, 0, 1, 0], dtype=np.byte)),
    ('0000', np.array([0, 0, 0, 0], dtype=np.byte)),
    ('1111', np.array([1, 1, 1, 1], dtype=np.byte)),
    ('10101010', np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.byte)),
])
def test_str_to_byte(string: str, bytes: npt.NDArray[np.byte]):
    assert (str_to_byte(string) == bytes).all()


@pytest.mark.parametrize('bytes, string', [
    (np.array([0], dtype=np.byte), '0'),
    (np.array([1], dtype=np.byte), '1'),
    (np.array([1, 0, 1, 0], dtype=np.byte), '1010'),
    (np.array([0, 0, 0, 0], dtype=np.byte), '0000'),
    (np.array([1, 1, 1, 1], dtype=np.byte), '1111'),
    (np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.byte), '10101010'),
])
def test_byte_to_str(bytes: npt.NDArray[np.byte], string: str):
    assert byte_to_str(bytes) == string


@pytest.mark.parametrize('dt, string', [
    (datetime(2022, 11, 7, 21, 11, 53), '2022_11_07_21-11-53'),
    (datetime(2023, 6, 26, 11, 13, 23), '2023_06_26_11-13-23'),
    (datetime(2024, 10, 6, 10, 21, 33), '2024_10_06_10-21-33'),
    (datetime(2021, 4, 11, 13, 37, 13), '2021_04_11_13-37-13'),
    (datetime(2011, 3, 26, 23, 8, 34), '2011_03_26_23-08-34'),
    (datetime(2013, 4, 30, 19, 1, 48), '2013_04_30_19-01-48'),
])
def test_timestamp_str(dt: datetime, string: str):
    assert timestamp_str(dt) == string


@pytest.mark.parametrize('origname, qubits, name', [
    ('fake_munich_v2', 20, 'Munich (20)'),
    ('fake_berlin_v2', 10, 'Berlin (10)'),
    ('fake_london_v2', 15, 'London (15)'),
    ('fake_prague_v2', 17, 'Prague (17)'),
    ('new_york_v2', 6, 'New York (6)'),
    ('fake_new_york', 64, 'New York (64)'),
    ('fake_new_york_v2', 10, 'New York (10)'),
])
def test_backend_name(origname: str, qubits: int, name: str):
    backend = Backend()
    backend.name = origname
    backend.num_qubits = qubits
    assert backend_name(backend) == name


@pytest.mark.parametrize('name, formatted', [
    ('one', 'One'),
    ('TWO', 'Two'),
    ('tHREE', 'Three'),
    ('fOuR', 'Four'),
    ('one_TWO', 'One Two'),
    ('THREE_four', 'Three Four'),
])
def test_method_name(name: str, formatted: str):
    assert method_name(name) == formatted


@pytest.mark.parametrize('size', [0, 1, 2, 3, 4, 10])
def test_all_measurement_outputs(size: int):
    outputs = all_measurement_outputs(size)
    assert len(set(outputs)) == 2 ** size


@pytest.mark.parametrize('dict1, dict2, size', [
    ({'1': 1}, {'0': 0, '1': 1}, 1),
    ({'0': 1}, {'0': 1, '1': 0}, 1),
    ({'10': 2}, {'00': 0, '10': 2, '01': 0, '11': 0}, 2),
])
def test_fill_counts(dict1: Dict[str, int], dict2: Dict[str, int], size: int):
    fill_counts(dict1, size)
    assert dict1 == dict2


@pytest.mark.parametrize('xvals, yvals, xvals2, yvals2', [
    ([3, 2, 1], [1, 2, 3], [1, 2, 3], [3, 2, 1]),
    ([3, 2, 1], [4, 5, 6], [1, 2, 3], [6, 5, 4]),
    ([1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]),
    ([3, 1, 2], [1, 2, 3], [1, 2, 3], [2, 3, 1]),
])
def test_sort_zipped(xvals: list[int], yvals: list[int], xvals2: list[int], yvals2: list[int]):
    xout, yout = sort_zipped(np.array(xvals), np.array(yvals))
    assert (xout == xvals2).all()
    assert (yout == yvals2).all()


@pytest.mark.parametrize('str1, str2, diff', [
    ('honzanammocchybi', 'honzanammocchybi', 0),
    ('honzanammocchybi', 'honzanamm1cchybi', 1),
    ('honzanammocchybi', 'honzana3mocchybi', 1),
    ('honzanammocchybi', 'honz4nammocchybi', 1),
    ('honzanammocchybi', 'honzanam12cchybi', 2),
    ('honzanammocchyb3', 'honzanam12cchybi', 3),
])
def test_diff_letters(str1: str, str2: str, diff: int):
    assert diff_letters(str1, str2) == diff


@pytest.mark.parametrize('secret, good', [
    ('0', True),
    ('1', True),
    ('', True),
    ('1000101', True),
    ('1101001', True),
    ('1111111', True),
    ('011' * 20, True),
    ('foo', False),
    ('bar', False),
    ('1001foo', False),
    ('bar10000', False),
])
def test_check_secret(secret: str, good: bool):
    assert check_secret(secret) == good


@pytest.mark.parametrize('arr, key, pos', [
    (['00', '01', '10', '11'], '00', 0),
    (['00', '01', '10', '11'], '01', 1),
    (['00', '01', '10', '11'], '10', 2),
    (['00', '01', '10', '11'], '11', 3),
    (['00', '01', '10', '11'], '110', 4),
    (['00', '01', '10', '11'], '111', 4),
    (['00', '01', '10', '11'], '0', 4),
])
def test_find_secret(arr: list[str], key: str, pos: int):
    assert find_secret(np.array(arr, dtype=str), key) == pos


@pytest.mark.parametrize('dictionary, key', [
    ({0: 0, 1: 1, 2: 2, 3: 3}, 3),
    ({0: 1, 1: 3, 2: 2, 3: 3}, 1),
    ({0: 2, 1: 5, 2: 2, 3: 3}, 1),
    ({0: 3, 1: 4, 2: 2, 3: 3}, 1),
    ({0: 4, 1: 1, 2: 5, 3: 3}, 2),
])
def test_dict_max_value_key(dictionary: Dict[int, int], key: int):
    assert dict_max_value_key(dictionary) == key

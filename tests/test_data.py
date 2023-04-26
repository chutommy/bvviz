from typing import Dict, Any
from unittest.mock import patch, mock_open, Mock

import pytest
from qiskit.providers import Backend

from src.bvviz.data import FmtStr, Descriptor, BackendDB


@pytest.mark.parametrize('string, kwargs, expected', [
    ('foo', {}, 'foo'),
    ('bar', {}, 'bar'),
    ('{name} nam moc chybi', {'name': 'honza'}, 'honza nam moc chybi'),
    ('foo {num} bar', {'num': 42}, 'foo 42 bar'),
])
def test_fmt_str(string: str, kwargs: Dict[str, Any], expected: str):
    out = FmtStr(string)(**kwargs)
    assert out == expected


def test_descriptor():
    with patch('builtins.open', mock_open(read_data='{"part1":"foo","part2":"bar"}')) as mock_file:
        descriptor = Descriptor('some/path')
        assert descriptor['part1']() == 'foo'
        assert descriptor['part2']() == 'bar'
        assert descriptor.cat(['part1', 'part2']) == 'foobar'
    mock_file.assert_called_with('some/path', 'r', encoding='utf-8')


@pytest.mark.parametrize('size', [0, 1, 2, 3, 10, 100, 1000])
def test_backend_db(size: int):
    backends = [Mock(spec=Backend) for _ in range(size)]
    database = BackendDB(backends)
    assert database.size() == size

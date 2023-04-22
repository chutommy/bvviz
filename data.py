import json
import re
import string
from typing import List

from qiskit.providers import Backend


class FmtStr:
    """FmtStr represents formattable helper string structure."""

    value: str = None

    def __init__(self, value: str):
        self.value = value

    def __call__(self, **kwargs):
        return self.value.format(**kwargs)

    def __str__(self):
        return self.value

    def __get__(self):
        return self.value


class Descriptor:
    """Descriptor handles long texts."""

    descriptions: dict = None

    def __init__(self):
        with open('assets/descriptions.json', 'r') as f:
            descriptions_json = f.read()
        self.descriptions = json.loads(descriptions_json)

    def __getitem__(self, item) -> str | FmtStr:
        args = re.findall(r'{(.*?)}', self.descriptions[item])
        if len(args) != 0:
            return FmtStr(self.descriptions[item])
        return self.descriptions[item]


class BackendDB:
    """Represents a database of accessible backends."""

    backends: List[Backend] = None

    def __init__(self, backends: List[Backend]):
        self.backends = list(backends)
        self.cap = len(backends)

    def __getitem__(self, key):
        return self.backends[key]

    def size(self) -> int:
        """Returns how many backends can be provided."""
        return self.cap

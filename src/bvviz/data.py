"""Handles regular data manipulation."""

from json import loads
from typing import List

from qiskit.providers import Backend


class FmtStr:
    """FmtStr represents format-friendly helper string structure."""

    def __init__(self, value: str):
        self.value = value

    def __call__(self, **kwargs) -> str:
        return self.value.format(**kwargs)

    def __str__(self) -> str:
        return self.value


class Descriptor:
    """Descriptor handles long texts."""

    def __init__(self, path: str):
        with open(path, 'r', encoding='utf-8') as file:
            descriptions_json = file.read()
        self.descriptions = loads(descriptions_json)

    def __getitem__(self, item) -> FmtStr:
        # args = findall(r'{(.*?)}', self.descriptions[item])
        # if len(args) != 0:
        #     return FmtStr(self.descriptions[item])
        # return self.descriptions[item]
        return FmtStr(self.descriptions[item])

    def cat(self, args: List[str]) -> str:
        """Concatenates multiple descriptions."""
        out = ''
        for arg in args:
            out += self.descriptions[arg]
        return out

    def __call__(self) -> dict:
        return self.descriptions


class BackendDB:
    """Represents a database of accessible backends."""

    def __init__(self, backends: List[Backend]):
        self.backends = list(backends)
        self.cap = len(backends)

    def __getitem__(self, key):
        return self.backends[key]

    def size(self) -> int:
        """Returns how many backends can be provided."""
        return self.cap

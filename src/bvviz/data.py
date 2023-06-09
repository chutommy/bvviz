"""This module implements helper functions and classes for docs data handling."""

from json import loads
from typing import Any, List, SupportsIndex

from qiskit.providers import Backend


class FmtStr:
    """Represents format-friendly helper string structure."""

    def __init__(self, value: str) -> None:
        self.value = value

    def __call__(self, **kwargs: str) -> str:
        if not kwargs:
            return self.value
        return self.value.format(**kwargs)

    def __str__(self) -> str:
        return self.value


class Descriptor:
    """Handles external string values."""

    def __init__(self, path: str) -> None:
        with open(path, 'r', encoding='utf-8') as file:
            descriptions_json = file.read()
        self.descriptions = loads(descriptions_json)

    def __getitem__(self, item: str) -> FmtStr:
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

    def __call__(self) -> Any:
        return self.descriptions


class BackendDB:
    """Represents a database of accessible backends."""

    def __init__(self, backends: List[Backend]) -> None:
        self.backends = list(backends)
        self.cap = len(backends)

    def __getitem__(self, key: SupportsIndex) -> Backend:
        return self.backends[key]

    def size(self) -> int:
        """Returns how many backends can be provided."""
        return self.cap

from abc import ABC
from functools import wraps

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister


def count_incrementer(method):
    """Increments the Oracle's query counter by one"""

    @wraps(method)
    def _impl(self, *method_args, **method_kwargs):
        self.query_count += 1
        return method(self, *method_args, **method_kwargs)

    return _impl


class Oracle(ABC):
    """A base oracle with unexposed operation."""

    def __init__(self, secret: np.array):
        self.complexity = len(secret)
        self.secret = secret
        self.query_count = 0


class ClassicalOracle(Oracle):
    """Represents an implementation of a classical oracle for the Bernstein-Vazirani problem."""

    @count_incrementer
    def query(self, inp: np.array) -> bool:
        """Apply the classical BV function."""
        product = np.dot(self.secret, inp)
        out_bit = product % 2
        return out_bit


class QuantumOracle(Oracle):
    """Represents an implementation of a quantum oracle for the Bernstein-Vazirani problem. The oracle is not queried
    like classical oracles, instead it is applied on a given quantum circuit with precisely specified operational
    quantum registers (input-output quantum bits)."""

    @count_incrementer
    def apply_circuit(self, circuit: QuantumCircuit, in_qreg: QuantumRegister, out_qreg: QuantumRegister):
        """Constructs an oracle within the given quantum circuit on top of the input/output qubits."""
        # correct size of quantum registers is essential for the correct implementation of the oracle
        if in_qreg.size != self.complexity:
            raise ValueError("Invalid input register size.")
        if out_qreg.size != 1:
            raise ValueError("Invalid output register size.")

        for i in range(self.complexity):
            if self.secret[self.complexity - i - 1]:
                circuit.cx(in_qreg[i], out_qreg[0])

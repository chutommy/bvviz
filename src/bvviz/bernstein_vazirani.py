"""This module implements classical and quantum implementations of Bernstein-Vazirani's algorithm.
The module contains both classical and quantum oracles."""

from dis import Bytecode
from functools import wraps
from random import random
from typing import Any, Callable, ParamSpec, TypeVar

import numpy as np
import numpy.typing as npt
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.quantum_info import random_statevector

Param = ParamSpec("Param")
RetType = TypeVar("RetType")


def count_incrementer(method: Callable[Param, RetType]) -> Any:
    """Increments the Oracles' query counter by one."""

    @wraps(method)
    def _impl(self: Any, *method_args: Any, **method_kwargs: Any) -> RetType:
        self.query_count += 1
        return method(self, *method_args, **method_kwargs)

    return _impl


class ClassicalOracle:
    """Represents an implementation of a classical oracle for the Bernstein-Vazirani problem."""

    def __init__(self, secret: npt.NDArray[np.byte]) -> None:
        self.complexity = len(secret)
        self.secret = secret
        self.query_count = 0

    def validate(self) -> bool:
        """Validate integrity of the oracle."""
        return self.secret is not None \
            and self.complexity == len(self.secret) \
            and self.query_count >= 0

    @count_incrementer
    def query(self, inp: npt.NDArray[np.byte]) -> Any:
        """Apply the classical BV function."""
        product = np.dot(self.secret, inp)
        out_bit = product % 2
        return out_bit


class QuantumOracle:
    """Represents an implementation of a quantum oracle for the Bernstein-Vazirani problem. The
    oracle is not queried like classical oracles, instead it is applied on a given quantum
    circuit with exactly specified operational quantum registers (input-output quantum bits)."""

    def __init__(self, secret: npt.NDArray[np.byte]) -> None:
        self.complexity = len(secret)
        self.secret = secret
        self.query_count = 0

    def validate(self) -> bool:
        """Validate integrity of the oracle."""
        return self.secret is not None \
            and self.complexity == len(self.secret) \
            and self.query_count >= 0

    def used(self) -> bool:
        """Checks whether the oracle was already used or not."""
        return self.query_count != 0

    @count_incrementer
    def apply_circuit(self, circuit: QuantumCircuit,
                      in_qreg: QuantumRegister,
                      out_qreg: QuantumRegister) -> None:
        """Constructs an oracle within the given quantum circuit."""
        # ensure correct size of quantum registers
        if in_qreg.size != self.complexity:
            raise ValueError('Invalid input register size.')
        if out_qreg.size != 1:
            raise ValueError('Invalid output register size.')

        for i in range(self.complexity):
            if self.secret[self.complexity - i - 1]:
                circuit.cx(in_qreg[i], out_qreg[0])


class ClassicalSolver:
    """Implements a classical solution to the BV problem."""

    @staticmethod
    def solve(oracle: ClassicalOracle) -> npt.NDArray[np.byte]:
        """Queries over all positional bits to determine the secret value of all indices."""
        solution = np.empty(oracle.complexity, dtype=np.byte)

        for i in range(oracle.complexity):
            # create a query with a single on-bit at exposing input index
            input_val = np.zeros(shape=oracle.complexity, dtype=np.byte)
            input_val[i] = 1

            # noinspection PyArgumentList
            answer = oracle.query(inp=input_val)
            if answer == 1:
                solution[i] = 1
            else:
                solution[i] = 0

        return solution

    def ops_count(self) -> int:
        """Calculates the number of instructions needed to solve the oracle.
        A line in disassembled solution code counts as one operation."""
        bytecode = Bytecode(self.solve)
        count = bytecode.dis().count('\n')
        return count


class QuantumCircuitBuild:
    """Represents a quantum circuit building tool for the implementation of the BV algorithm."""

    def __init__(self) -> None:
        self.qreg = QuantumRegister(1)
        self.creg = ClassicalRegister(1)
        self.auxreg = QuantumRegister(1)
        self.circuit = QuantumCircuit()

    def allocate_registers(self, complexity: int) -> None:
        """Allocates quantum and classical register according to oracle's complexity."""
        self.qreg = QuantumRegister(size=complexity, name='qreg')
        self.creg = ClassicalRegister(size=complexity, name='creg')
        self.auxreg = QuantumRegister(size=1, name='auxreg')
        self.circuit = QuantumCircuit(self.qreg, self.auxreg, self.creg,
                                      name='cirq', global_phase=random() * np.pi)

    # noinspection PyUnresolvedReferences
    def simulate_random_initial_state(self) -> None:
        """Initializes quantum registers at random states."""
        # https://quantumcomputing.stackexchange.com/q/4962
        # pylint: disable=no-member
        for qubit in self.qreg:
            self.circuit.initialize(random_statevector(2).data, qubit)
        self.circuit.initialize(random_statevector(2), self.auxreg)

    def reset_registers(self) -> None:
        """Introduces quantum registers into ket zeroes."""
        for qubit in self.qreg:
            self.circuit.reset(qubit)
        self.circuit.reset(self.auxreg)

    def prepare_initial_state(self) -> None:
        """Prepares initial state of all quantum qubits"""
        # introduce qubits into superposition
        for qubit in self.qreg:
            self.circuit.h(qubit)

        # prepare auxiliary qubit to eigenvector of eigenvalue -1 of the CX gate
        # https://en.wikipedia.org/wiki/Controlled_NOT_gate
        # self.circuit.h(self.auxreg)
        # self.circuit.z(self.auxreg)
        self.circuit.x(self.auxreg)
        self.circuit.h(self.auxreg)

    def measure(self) -> None:
        """Apply measurement of quantum query register on the classical register."""
        for qubit in self.qreg:
            self.circuit.h(qubit)

        self.circuit.measure(self.qreg, self.creg)

    def create_circuit(self, oracle: QuantumOracle, random_initialization: bool) -> None:
        """Builds a quantum implementation of the BV algorithm with the preset secret."""
        self.allocate_registers(complexity=oracle.complexity)
        if random_initialization:
            self.simulate_random_initial_state()
        self.reset_registers()
        self.prepare_initial_state()

        self.circuit.barrier()
        # noinspection PyArgumentList
        oracle.apply_circuit(circuit=self.circuit,
                             in_qreg=self.qreg,
                             out_qreg=self.auxreg)
        self.circuit.barrier()

        self.measure()

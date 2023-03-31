from abc import ABC
from functools import wraps

import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.quantum_info import random_statevector
from qiskit_aer.noise import NoiseModel, pauli_error, depolarizing_error, reset_error


def count_incrementer(method):
    """Increments the Oracles' query counter by one."""

    @wraps(method)
    def _impl(self, *method_args, **method_kwargs):
        self.query_count += 1
        return method(self, *method_args, **method_kwargs)

    return _impl


class Oracle(ABC):
    """A base oracle with unexposed function."""

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
    like classical oracles, instead it is applied on a given quantum circuit with exactly specified operational
    quantum registers (input-output quantum bits)."""

    @count_incrementer
    def apply_circuit(self, circuit: QuantumCircuit, in_qreg: QuantumRegister, out_qreg: QuantumRegister):
        """Constructs an oracle within the given quantum circuit on top of the input/output qubits."""
        # ensure correct size of quantum registers
        if in_qreg.size != self.complexity:
            raise ValueError("Invalid input register size.")
        if out_qreg.size != 1:
            raise ValueError("Invalid output register size.")

        for i in range(self.complexity):
            if self.secret[self.complexity - i - 1]:
                circuit.cx(in_qreg[i], out_qreg[0])


class ClassicalSolver:
    """Implements a classical solution to the BV problem."""

    def __init__(self, oracle: ClassicalOracle):
        self.oracle = oracle

    def solve(self) -> np.array:
        """Queries over all positional bits to determine the secret value of all indices."""
        solution = np.empty(self.oracle.complexity, dtype=np.byte)

        for i in range(self.oracle.complexity):
            # create a query with a single on-bit at exposing input index
            input_val = np.zeros(self.oracle.complexity, dtype=np.byte)
            input_val[i] = 1

            answer = self.oracle.query(input_val)
            if answer == 1:
                solution[i] = 1
            else:
                solution[i] = 0

        return solution


class QuantumCircuitBuild:
    """Represents a quantum circuit building tool for the implementation of the Bernstein-Vazirani algorithm."""

    def __init__(self, oracle: QuantumOracle):
        self.oracle = oracle

        self.qreg = None  # quantum BV query register
        self.auxreg = None  # auxiliary output qubit register
        self.creg = None  # classical measurement register

        self.circuit = None

    def allocate_registers(self):
        """Allocates quantum and classical register according to oracle's complexity."""
        self.qreg = QuantumRegister(self.oracle.complexity, "qreg")
        self.creg = ClassicalRegister(self.oracle.complexity, "creg")
        self.auxreg = QuantumRegister(1, "auxreg")
        self.circuit = QuantumCircuit(self.qreg, self.auxreg, self.creg)

    def __simulate_random_initial_state(self):
        """Initializes quantum registers at random states."""
        for qubit in self.qreg:
            self.circuit.initialize(random_statevector(2).data, qubit)
        self.circuit.initialize(random_statevector(2), self.auxreg)

    def reset_registers(self):
        """Introduces quantum registers into ket zeroes."""
        for qubit in self.qreg:
            self.circuit.reset(qubit)
        self.circuit.reset(self.auxreg)

    def prepare_initial_state(self):
        """Prepares initial state of all quantum qubits"""
        # introduce qubits into superposition
        for qubit in self.qreg:
            self.circuit.h(qubit)

        # prepare auxiliary qubit to eigenvector of eigenvalue -1 of the CX gate
        # https://en.wikipedia.org/wiki/Controlled_NOT_gate
        self.circuit.x(self.auxreg)
        self.circuit.h(self.auxreg)
        # self.circuit.h(self.auxreg)
        # self.circuit.z(self.auxreg)

    def measure(self):
        """Apply measurement of quantum query register on the classical register."""
        for qubit in self.qreg:
            self.circuit.h(qubit)

        self.circuit.measure(self.qreg, self.creg)

    def create_circuit(self):
        """Builds a quantum implementation of the Bernstein-Vazirani's algorithm with the preset secret."""
        self.allocate_registers()
        self.__simulate_random_initial_state()
        self.reset_registers()
        self.prepare_initial_state()

        self.circuit.barrier()
        self.oracle.apply_circuit(self.circuit, self.qreg, self.auxreg)
        self.circuit.barrier()

        self.measure()


class NoiseBuild:
    """Builds a simple custom noise model to imitate real quantum computer."""

    def __init__(self):
        self.model = NoiseModel()

    def applyResetError(self, rate: float = 0.03):
        """Applies reset error channel."""
        error_reset = reset_error(rate, 1 - rate)
        self.model.add_all_qubit_quantum_error(error_reset, "reset")

    def applyMeasurementError(self, rate: float = 0.05):
        """Applies measurement error channel."""
        error_meas = pauli_error([('X', rate), ('I', 1 - rate)])
        self.model.add_all_qubit_quantum_error(error_meas, "measure")

    def applyGateError(self, single_rate: float = 0.07, double_rate: float = 0.11):
        """Applies gates' error channels."""
        error_1 = depolarizing_error(single_rate, 1)
        self.model.add_all_qubit_quantum_error(error_1, ["h", "z", "x"])

        error_2 = depolarizing_error(double_rate, 2)
        self.model.add_all_qubit_quantum_error(error_2, ["cx"])

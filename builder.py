import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

from oracle import ClassicalOracle, QuantumOracle


class ClassicalSolver:
    """Implements a classical solution to the BV problem."""

    def __init__(self, oracle: ClassicalOracle):
        self.oracle = oracle

    def solve(self) -> np.array:
        """Queries over all positional bits to determine the secret value of all indices."""
        n = self.oracle.complexity
        solution = np.empty(n, dtype=np.byte)

        for i in range(n):
            # create a query with a single on-bit at exposing input index
            input_val = np.zeros(n, dtype=np.byte)
            input_val[i] = 1

            answer = self.oracle.query(input_val)
            if answer == 1:
                solution[i] = 1
            else:
                solution[i] = 0

        return solution


class BVAlgBuilder:
    """Represents a quantum circuit building tool for the implementation of the Bernstein-Vazirani algorithm."""

    def __init__(self, oracle: QuantumOracle):
        self.oracle = oracle

        self.qreg = None  # quantum register for querying
        self.auxreg = None  # auxiliary register for output qubit
        self.creg = None  # classical register for measurement

        self.circuit = None

    def __allocate_registers(self):
        self.qreg = QuantumRegister(self.oracle.complexity, "qreg")
        self.creg = ClassicalRegister(self.oracle.complexity, "creg")
        self.auxreg = QuantumRegister(1, "auxreg")
        self.circuit = QuantumCircuit(self.qreg, self.auxreg, self.creg)

    def __prepare_initial_state(self):
        # introduce qubits into superposition
        for qubit in self.qreg:
            self.circuit.h(qubit)

        # prepare auxiliary qubit to eigenvector of the CX gate
        self.circuit.x(self.auxreg)
        self.circuit.h(self.auxreg)
        # self.circuit.h(self.auxreg)
        # self.circuit.z(self.auxreg)

    def __measure(self):
        for qubit in self.qreg:
            self.circuit.h(qubit)

        self.circuit.measure(self.qreg, self.creg)

    def create_circuit(self):
        self.__allocate_registers()
        self.__prepare_initial_state()

        self.circuit.barrier()
        self.oracle.apply_circuit(self.circuit, self.qreg, self.auxreg)
        self.circuit.barrier()

        self.__measure()

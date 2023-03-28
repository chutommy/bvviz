import numpy as np
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister


class ClassicalOracle:
    def __init__(self, secret: np.array):
        self.complexity = len(secret)
        self.secret = secret
        self.query_count = 0

    def query(self, inp: np.array) -> bool:
        if self.secret is None or self.complexity is None:
            raise ValueError("Secret has not been specified.")

        self.query_count += 1

        product = np.dot(self.secret, inp)
        out_bit = product % 2

        return out_bit


class ClassicalSolver:
    def __init__(self, oracle: ClassicalOracle):
        self.oracle = oracle

    def solve(self) -> np.array:
        n = self.oracle.complexity
        solution = np.empty(n, dtype=np.byte)

        for i in range(n):
            inp = np.zeros(n, dtype=np.byte)
            inp[i] = 1

            if self.oracle.query(inp):
                solution[i] = 1
            else:
                solution[i] = 0

        return solution


class QuantumOracle:
    def __init__(self, secret: np.array):
        self.complexity = len(secret)
        self.secret = secret
        self.query_count = 0

    def apply_circuit(self, circuit: QuantumCircuit, in_qreg: QuantumRegister, out_qreg: QuantumRegister):
        if self.secret is None or self.complexity is None:
            raise ValueError("Secret has not been specified.")

        if in_qreg.size != self.complexity:
            raise ValueError("Invalid input register size.")
        if out_qreg.size != 1:
            raise ValueError("Invalid output register size.")

        self.query_count += 1

        for i in range(self.complexity):
            if self.secret[self.complexity - i - 1]:
                circuit.cx(in_qreg[i], out_qreg[0])


class BVAlgBuilder:
    def __init__(self, oracle: QuantumOracle):
        self.oracle = oracle

        self.qreg = None
        self.auxreg = None
        self.creg = None
        self.circuit = None

    def alloc_registers(self):
        self.qreg = QuantumRegister(self.oracle.complexity, "qreg")
        self.creg = ClassicalRegister(self.oracle.complexity, "creg")
        self.auxreg = QuantumRegister(1, "auxreg")
        self.circuit = QuantumCircuit(self.qreg, self.auxreg, self.creg)

    def prep_superposition(self):
        for qubit in self.qreg:
            self.circuit.h(qubit)

        # prepare auxiliary qubit to eigenvector of the CX gate
        self.circuit.x(self.auxreg)
        self.circuit.h(self.auxreg)

    def measure(self):
        for qubit in self.qreg:
            self.circuit.h(qubit)

        self.circuit.measure(self.qreg, self.creg)

    def create_circuit(self):
        self.alloc_registers()
        self.prep_superposition()
        self.circuit.barrier()
        self.oracle.apply_circuit(self.circuit, self.qreg, self.auxreg)
        self.circuit.barrier()
        self.measure()

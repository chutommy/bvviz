import numpy as np
import pytest
from qiskit import QuantumCircuit, QuantumRegister

from src.bvviz.bernstein_vazirani import (ClassicalOracle, ClassicalSolver, QuantumCircuitBuild,
                                          QuantumOracle)


@pytest.mark.parametrize('count', [0, 1, 2, 3, 10, 100, 1000])
def test_count_incrementer(count: int):
    oracle = ClassicalOracle(np.array([1, 0, 1, 0], dtype=np.byte))
    for _ in range(count):
        oracle.query(np.array([0, 0, 0, 0], dtype=np.byte))
    assert oracle.query_count == count


@pytest.mark.parametrize('oracle', [
    ClassicalOracle(np.array([1, 0, 1, 0], dtype=np.byte)),
    QuantumOracle(np.array([1, 0, 1, 0], dtype=np.byte)),
])
def test_validate(oracle):
    size = len(oracle.secret)
    assert oracle.validate()
    oracle.complexity = -1
    assert not oracle.validate()
    oracle.complexity = size

    assert oracle.validate()
    oracle.query_count = -1
    assert not oracle.validate()
    oracle.query_count = 0

    assert oracle.validate()
    oracle.secret = None
    assert not oracle.validate()


@pytest.mark.parametrize('secret, query, output', [
    ('1010', '1000', 1),
    ('1010', '0100', 0),
    ('1010', '0010', 1),
    ('1010', '0001', 0),
    ('1010', '1100', 1),
    ('1010', '1010', 0),
    ('1010', '1001', 1),
    ('1010', '1110', 0),
    ('1010', '1101', 1),
    ('1010', '1111', 0),
])
def test_cl_query(secret: str, query: str, output: int):
    oracle = ClassicalOracle(np.array(list(secret), dtype=np.byte))
    assert oracle.query(inp=np.array(list(query), dtype=np.byte)) == output


def test_qu_used():
    oracle = QuantumOracle(np.array([1, 0, 1, 0], dtype=np.byte))
    assert not oracle.used()
    oracle.query_count = 1
    assert oracle.used()


@pytest.mark.parametrize('secret, inq, outq, noerr', [
    ('1', 1, 1, True),
    ('10', 2, 1, True),
    ('1010', 4, 1, True),
    ('10101', 5, 1, True),
    ('111111', 6, 1, True),
    ('1010', 3, 1, False),
    ('1010', 5, 1, False),
    ('1010', 4, 0, False),
    ('1010', 4, 2, False),
])
def test_apply_circuit(secret: str, inq: int, outq: int, noerr: bool):
    oracle = QuantumOracle(np.array(list(secret), dtype=np.byte))
    qreg = QuantumRegister(inq)
    auxreg = QuantumRegister(outq)
    circ = QuantumCircuit(qreg, auxreg)
    if noerr:
        try:
            oracle.apply_circuit(circuit=circ, in_qreg=qreg, out_qreg=auxreg)
        except ValueError as exc:
            assert False, exc
    else:
        with pytest.raises(Exception):
            oracle.apply_circuit(circ, qreg, auxreg)


@pytest.mark.parametrize('secret', ['00', '10', '11', '0000', '1010',
                                    '1111', '0000', '10101', '11111'])
def test_classical_solver(secret: str, ):
    solver = ClassicalSolver()
    secret_seq = np.array(list(secret), dtype=np.byte)
    oracle = ClassicalOracle(secret_seq)
    assert (solver.solve(oracle) == secret_seq).all()
    assert solver.ops_count() > 0


def test_create_circuit():
    oracle = QuantumOracle(np.array([1, 0, 1, 0], dtype=np.byte))
    builder = QuantumCircuitBuild()
    try:
        builder.create_circuit(oracle, True)
    except ValueError as exc:
        assert False, exc

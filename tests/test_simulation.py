from random import randint

import pytest
from qiskit import QuantumCircuit, QuantumRegister

from src.bvviz.config import NoiseConfiguration, TranspileConfiguration
from src.bvviz.simulation import NoiseConfig, Simulator, BackendService


# noinspection PyArgumentList
@pytest.mark.parametrize('rate, good', [
    (0, True),
    (1, True),
    (0.1324, True),
    (0.9421, True),
    (0.112, True),
    (0.423, True),
    (-0.134, False),
    (-3.123, False),
    (1.134, False),
    (5.1432, False),
])
def test_noise_config(rate: float, good: bool):
    config = NoiseConfig()
    methods = [
        config.apply_reset_error,
        config.apply_measurement_error,
        config.apply_single_gate_error,
        config.apply_double_gate_error,
    ]
    for method in methods:
        if good:
            try:
                method(rate)
            except ValueError as exception:
                assert False, exception
        else:
            with pytest.raises(Exception):
                method(rate)


@pytest.mark.parametrize('complexity, shots, seed', [
    (6, 6, 6),
    (1, 2, 3),
    (3, 10, randint(1, 10 ** 10)),
    (5, 10, randint(1, 10 ** 10)),
])
def test_simulator(complexity: int, shots: int, seed: int):
    sim = Simulator()
    # pylint: disable=W0703
    try:
        sim.set_noise(NoiseConfiguration())
        sim.transpile(QuantumCircuit(QuantumRegister(complexity)), TranspileConfiguration(), seed)
        sim.execute(shots, seed)
    except Exception as exc:
        assert False, exc


def test_backend_service():
    src = BackendService()
    assert len(src.list_backends()) != 0
    # pylint: disable=W0703
    try:
        backend = src.get_backend()
        assert backend is not None
    except Exception as exc:
        assert False, exc

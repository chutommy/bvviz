"""Provides custom simulators and simulation services."""

from typing import Any

from qiskit import QuantumCircuit, transpile
from qiskit.providers import Backend, Job
from qiskit.providers.fake_provider import FakeProviderForBackendV2, FakeQasmSimulator
from qiskit_aer.noise import depolarizing_error, NoiseModel, pauli_error, reset_error

from .config import NoiseConfiguration, TranslationMethod, TranspileConfiguration


class NoiseConfig:
    """Builds a simple custom noise model to imitate real quantum computer."""

    def __init__(self) -> None:
        self.model = NoiseModel()

    def apply_reset_error(self, rate: float) -> None:
        """Applies reset error channel."""
        if rate < 0 or rate > 1:
            raise ValueError('invalid error rate')
        error_reset = reset_error(1 - rate, rate)
        self.model.add_all_qubit_quantum_error(error_reset, 'reset')

    def apply_measurement_error(self, rate: float) -> None:
        """Applies measurement error channel."""
        if rate < 0 or rate > 1:
            raise ValueError('invalid error rate')
        error_meas = pauli_error([('X', rate), ('I', 1 - rate)])
        self.model.add_all_qubit_quantum_error(error_meas, 'measure')

    def apply_single_gate_error(self, rate: float) -> None:
        """Applies single gate error channel."""
        if rate < 0 or rate > 1:
            raise ValueError('invalid error rate')
        error = depolarizing_error(rate, 1)
        self.model.add_all_qubit_quantum_error(error, ['i', 'h', 'z', 'x'])

    def apply_double_gate_error(self, rate: float) -> None:
        """Applies double gate error channel."""
        if rate < 0 or rate > 1:
            raise ValueError('invalid error rate')
        error = depolarizing_error(rate, 2)
        self.model.add_all_qubit_quantum_error(error, ['cx'])


class Simulator:
    """Represents a wrapper for the providers' Aer backend simulation providers."""

    def __init__(self) -> None:
        self.noise_config: NoiseConfig = NoiseConfig()
        self.backend: Backend = FakeQasmSimulator()
        self.compiled_circuit: QuantumCircuit = QuantumCircuit()

    def set_noise(self, config: NoiseConfiguration) -> None:
        """Applies noise channels."""
        if config.reset_rate:
            self.noise_config.apply_reset_error(rate=config.reset_rate)

        if config.measure_rate:
            self.noise_config.apply_measurement_error(rate=config.measure_rate)

        if config.single_gate_rate:
            self.noise_config.apply_single_gate_error(rate=config.single_gate_rate)

        if config.double_gate_rate:
            self.noise_config.apply_double_gate_error(rate=config.double_gate_rate)

    def set_backend(self, backend: Backend) -> None:
        """Set custom backend."""
        self.backend = backend

    def transpile(self, circuit: QuantumCircuit, config: TranspileConfiguration, seed: int) -> None:
        """Return a compiled quantum circuit on the configured backend provider."""
        self.compiled_circuit = transpile(circuits=circuit,
                                          backend=self.backend,
                                          seed_transpiler=seed,
                                          basis_gates=self.noise_config.model.basis_gates,
                                          layout_method=config.layout_method,
                                          routing_method=config.routing_method,
                                          translation_method=TranslationMethod.SYNTHESIS.value,
                                          approximation_degree=config.approximation_degree,
                                          optimization_level=config.optimization_level)

    def execute(self, shots: int, seed_simulator: int,
                compiled_circuit: QuantumCircuit = None) -> Job:
        """Runs the compiled circuit."""
        if compiled_circuit is not None:
            self.compiled_circuit = compiled_circuit

        # noinspection PyUnresolvedReferences
        return self.backend.run(run_input=self.compiled_circuit,
                                shots=shots,
                                noise_model=self.noise_config.model,
                                memory=True,
                                seed_simulator=seed_simulator,
                                init_qubits=False)


# https://qiskit.org/documentation/apidoc/providers_fake_provider.html
class BackendService:
    """Backend service provider for builds to mimic the behavior of IBM Quantum systems."""

    def __init__(self) -> None:
        self.provider = FakeProviderForBackendV2()
        self.backend = FakeQasmSimulator()

    def list_backends(self) -> Any:
        """Returns available list of provided backends."""
        return self.provider.backends()

    def get_backend(self) -> Backend:
        """Returns service's default quantum backend."""
        return self.backend

"""Provides custom backend simulators and services."""

from qiskit.providers import Backend
from qiskit_aer.noise import NoiseModel, pauli_error, depolarizing_error, reset_error


class NoiseConfig:
    """Builds a simple custom noise model to imitate real quantum computer."""

    def __init__(self):
        self.model = NoiseModel()

    def apply_reset_error(self, rate: float):
        """Applies reset error channel."""
        error_reset = reset_error(rate, 1 - rate)
        self.model.add_all_qubit_quantum_error(error_reset, "reset")

    def apply_measurement_error(self, rate: float):
        """Applies measurement error channel."""
        error_meas = pauli_error([('X', rate), ('I', 1 - rate)])
        self.model.add_all_qubit_quantum_error(error_meas, "measure")

    def apply_single_gate_error(self, rate: float):
        """Applies single gate error channel."""
        error = depolarizing_error(rate, 1)
        self.model.add_all_qubit_quantum_error(error, ["h", "z", "x"])

    def apply_double_gate_error(self, rate: float):
        """Applies double gate error channel."""
        error = depolarizing_error(rate, 2)
        self.model.add_all_qubit_quantum_error(error, ["cx"])


class Simulator:
    """Represents a wrapper for the providers' Aer backend simulation providers."""

    def __init__(self):
        self.backend = None
        self.noise_config = None

    def set_noise(self, reset_rate: float = None, measure_rate: float = None,
                  single_gate_rate: float = None, double_gate_rate: float = None):
        """Applies noise channels."""
        self.noise_config = NoiseConfig()

        if reset_rate:
            self.noise_config.apply_reset_error(rate=reset_rate)
        if measure_rate:
            self.noise_config.apply_measurement_error(rate=measure_rate)
        if single_gate_rate:
            self.noise_config.apply_single_gate_error(rate=single_gate_rate)
        if double_gate_rate:
            self.noise_config.apply_double_gate_error(rate=double_gate_rate)

    def set_backend(self, backed: Backend):
        """Set custom backend."""
        self.backend = backed

"""This module provides configuration for the app engine."""

from dataclasses import dataclass
from enum import Enum

from qiskit.providers.fake_provider import FakeQasmSimulator


# https://qiskit.org/documentation/apidoc/transpiler_passes.html#layout-selection-placement
class LayoutMethod(Enum):
    """Name of the layout selection pass method."""

    TRIVIAL = 'trivial'
    DENSE = 'dense'
    NOISE_ADAPTIVE = 'noise_adaptive'
    SABRE = 'sabre'


# https://qiskit.org/documentation/apidoc/transpiler_passes.html#routing
class RoutingMethod(Enum):
    """Name of the routing pass method."""

    BASIC = 'basic'
    LOOKAHEAD = 'lookahead'
    STOCHASTIC = 'stochastic'
    SABRE = 'sabre'  # NONE = 'none'


# https://qiskit.org/documentation/apidoc/transpiler_passes.html#basis-change
class TranslationMethod(Enum):
    """Name of the translation pass method."""

    UNROLLER = 'unroller'
    TRANSLATOR = 'translator'
    SYNTHESIS = 'synthesis'


class OptimizationLevel(Enum):
    """The level of the circuit optimization."""

    NO = 0
    LIGHT = 1
    HEAVY = 2
    EVEN_HEAVIER = 3


@dataclass
class NoiseConfiguration:
    """Configuration of the noise model."""

    reset_rate: float = 0
    measure_rate: float = 0
    single_gate_rate: float = 0
    double_gate_rate: float = 0


@dataclass
class TranspileConfiguration:
    """Configuration of the transpiler."""

    layout_method: str = LayoutMethod.TRIVIAL.value
    routing_method: str = RoutingMethod.BASIC.value
    translation_method: str = TranslationMethod.TRANSLATOR.value

    approximation_degree: float = 1
    optimization_level: int = OptimizationLevel.NO.value


class Configuration:
    """Configuration of the Bernstein-Vazirani protocol simulator."""

    def __init__(self) -> None:
        self.transpiler_seed = 42
        self.simulator_seed = 42
        self.shot_count = 1000
        self.backend = FakeQasmSimulator()
        self.noise_config = NoiseConfiguration()
        self.transpile_config = TranspileConfiguration()

    def reset_partial(self) -> None:
        """Resets callable configuration settings."""
        self.backend = FakeQasmSimulator()
        self.noise_config = NoiseConfiguration()
        self.transpile_config = TranspileConfiguration()

    def reset(self) -> None:
        """Resets configuration."""
        self.transpiler_seed = 1
        self.simulator_seed = 1
        self.shot_count = 1000
        self.reset_partial()

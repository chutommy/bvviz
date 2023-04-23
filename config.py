"""Provides configuration for the app engine."""

from dataclasses import dataclass
from enum import Enum

import qiskit.providers.fake_provider as provider


# https://qiskit.org/documentation/apidoc/transpiler_passes.html#layout-selection-placement
class LayoutMethod(Enum):
    """Name of layout selection pass method."""

    TRIVIAL = "trivial"
    DENSE = "dense"
    NOISE_ADAPTIVE = "noise_adaptive"
    SABRE = "sabre"


# https://qiskit.org/documentation/apidoc/transpiler_passes.html#routing
class RoutingMethod(Enum):
    """Name of routing pass method."""

    BASIC = "basic"
    LOOKAHEAD = "lookahead"
    STOCHASTIC = "stochastic"
    SABRE = "sabre"
    # NONE = "none"


# https://qiskit.org/documentation/apidoc/transpiler_passes.html#basis-change
class TranslationMethod(Enum):
    """Name of translation pass method."""

    UNROLLER = "unroller"
    TRANSLATOR = "translator"
    SYNTHESIS = "synthesis"


class OptimizationLevel(Enum):
    """The level of optimized circuits."""

    NO = 0
    LIGHT = 1
    HEAVY = 2
    EVEN_HEAVIER = 3


@dataclass
class NoiseConfiguration:
    """Configuration for the noise model."""

    reset_rate: float = 0
    measure_rate: float = 0
    single_gate_rate: float = 0
    double_gate_rate: float = 0


@dataclass
class TranspileConfiguration:
    """Configuration for the transpiler."""

    layout_method: str = LayoutMethod.TRIVIAL
    routing_method: str = RoutingMethod.BASIC
    translation_method: str = TranslationMethod.TRANSLATOR

    approximation_degree: float = 1
    optimization_level: int = OptimizationLevel.NO


class Configuration:
    """Configuration for the Bernstein-Vazirani protocol simulator."""

    def reset_partial(self):
        """Reset callable configuration settings."""
        self.backend = provider.FakeQasmSimulator()
        self.noise_config = NoiseConfiguration()
        self.transpile_config = TranspileConfiguration()

    def __init__(self):
        self.transpiler_seed = 42
        self.simulator_seed = 42
        self.shot_count = 1000
        self.reset_partial()

    def reset(self):
        """Resets configuration."""
        self.transpiler_seed = 0
        self.simulator_seed = 0
        self.shot_count = 1000
        self.reset_partial()

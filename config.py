"""Provides configuration for the simulator."""
from dataclasses import dataclass
from enum import Enum

import qiskit.providers.fake_provider as provider
from qiskit.providers import Backend


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
    NONE = "none"


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

    layout_method: LayoutMethod = LayoutMethod.TRIVIAL
    routing_method: RoutingMethod = RoutingMethod.NONE
    translation_method: TranslationMethod = TranslationMethod.UNROLLER

    approximation_degree: float = 1
    optimization_level: OptimizationLevel = OptimizationLevel.NO


@dataclass
class Configuration:
    """Configuration for the Bernstein-Vazirani protocol simulator."""

    seed: int = 42
    shot_count: int = 1000

    backend: Backend = provider.FakeQasmSimulator()
    noise_config: NoiseConfiguration = NoiseConfiguration()
    transpile_config: TranspileConfiguration = TranspileConfiguration()

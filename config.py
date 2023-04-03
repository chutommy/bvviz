"""Provides configuration for the simulator."""
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


# https://qiskit.org/documentation/apidoc/providers_fake_provider.html
class Backend(Enum):
    """Fake backends built to mimic the behavior of IBM Quantum systems using system snapshots."""

    ALMADEN = provider.FakeAlmadenV2()
    ARMONK = provider.FakeArmonkV2()
    ATHENS = provider.FakeAthensV2()
    AUCKLAND = provider.FakeAuckland()
    BELEM = provider.FakeBelemV2()
    BOEBLINGEN = provider.FakeBoeblingenV2()
    BOGOTA = provider.FakeBogotaV2()
    BROOKLYN = provider.FakeBrooklynV2()
    BURLINGTON = provider.FakeBurlingtonV2()
    CAIRO = provider.FakeCairoV2()
    CAMBRIDGE = provider.FakeCambridgeV2()
    ESSEX = provider.FakeEssexV2()
    GENEVA = provider.FakeGeneva()
    GUADALUPE = provider.FakeGuadalupeV2()
    HANOI = provider.FakeHanoiV2()
    JAKARTA = provider.FakeJakartaV2()
    JOHANNESBURG = provider.FakeJohannesburgV2()
    KOLKATA = provider.FakeKolkataV2()
    LAGOS = provider.FakeLagosV2()
    LIMA = provider.FakeLimaV2()
    LONDON = provider.FakeLondonV2()
    MANHATTAN = provider.FakeManhattanV2()
    MELBOURNE = provider.FakeMelbourneV2()
    MONTREAL = provider.FakeMontrealV2()
    MUMBAI = provider.FakeMumbaiV2()
    NAIROBI = provider.FakeNairobiV2()
    OSLO = provider.FakeOslo()
    OURENSE = provider.FakeOurenseV2()
    PARIS = provider.FakeParisV2()
    PRAGUE = provider.FakePrague()
    POUGHKEEPSIE = provider.FakePoughkeepsieV2()
    ROCHESTER = provider.FakeRochesterV2()
    SANTIAGO = provider.FakeSantiagoV2()
    SINGAPORE = provider.FakeSingaporeV2()
    TORONTO = provider.FakeTorontoV2()
    VALENCIA = provider.FakeValenciaV2()
    VIGO = provider.FakeVigoV2()
    WASHINGTON = provider.FakeWashingtonV2()
    YORKTOWN = provider.FakeYorktownV2()


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
    backend: Backend = provider.FakeQasmSimulator()
    shot_count: int = 1000

    noise_config: NoiseConfiguration = NoiseConfiguration()
    transpile_config: TranspileConfiguration = TranspileConfiguration()

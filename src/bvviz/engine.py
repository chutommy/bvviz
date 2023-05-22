"""This module implements the quantum experiment engine."""

from dataclasses import dataclass
from json import dumps
from time import perf_counter_ns
from typing import Any, Dict, List, Type

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch, Patch
from matplotlib.ticker import MaxNLocator
from qiskit.providers import Job
from qiskit.result import Result as QiResult
from qiskit.visualization import plot_circuit_layout, plot_error_map, plot_gate_map

from .bernstein_vazirani import ClassicalOracle, ClassicalSolver, QuantumCircuitBuild, QuantumOracle
from .config import Configuration
from .data import BackendDB
from .simulation import BackendService, Simulator
from .utils import (byte_to_str, dict_max_value_key, diff_letters, fill_counts, find_secret,
                    generate_seed, pct_to_str, sort_zipped, str_to_byte, timestamp_str)


@dataclass
class CLResult:
    """Represents a classical solution."""

    oracle: ClassicalOracle
    solution: str
    time: float


@dataclass
class QUResult:
    """Represents a quantum solution."""

    oracle: QuantumOracle
    solution: str
    time: float


@dataclass
class EngineSnapshot:
    """Represents configuration and setup of the experiment."""

    configuration: Configuration
    backend_src: BackendService
    backend_db: BackendDB
    solver: ClassicalSolver
    builder: QuantumCircuitBuild
    sim: Simulator


@dataclass
class Result:
    """Is an output of an experiment. It stores a snapshot of the experiment."""

    secret: str

    job = Job()
    measurements: List[str]
    counts: Dict[str, int]
    result: QiResult

    cl_result: Type[CLResult] = CLResult
    qu_result: Type[QUResult] = QUResult
    snap: Type[EngineSnapshot] = EngineSnapshot


class Engine:
    """Handles incoming experiments."""

    def __init__(self) -> None:
        self.configuration = Configuration()
        self.backend_src = BackendService()
        self.backend_db = BackendDB(self.backend_src.list_backends())
        self.solver = ClassicalSolver()
        self.builder = QuantumCircuitBuild()
        self.sim = Simulator()

    def configure(self, config: Dict[str, Any]) -> None:
        """Configures current experiment."""
        self.configuration.backend = self.backend_db[config['backend_choice']]
        self.configuration.shot_count = config['shots']
        self.configuration.simulator_seed = generate_seed()
        self.configuration.transpiler_seed = generate_seed()

        self.configuration.noise_config.reset_rate = config['reset_err']
        self.configuration.noise_config.measure_rate = config['meas_err']
        self.configuration.noise_config.single_gate_rate = config['single_err']
        self.configuration.noise_config.double_gate_rate = config['double_err']

        self.configuration.transpile_config.layout_method = config['layout']
        self.configuration.transpile_config.routing_method = config['routing']
        self.configuration.transpile_config.translation_method = config['translation']
        self.configuration.transpile_config.optimization_level = config['optimization']
        self.configuration.transpile_config.approximation_degree = config['approx']

    def check_secret_size(self, secret: str) -> bool:
        """Verifies that secret is of a correct size."""
        size = len(secret)
        return size <= self.configuration.backend.num_qubits - 1 and size != 0

    def run(self, secret_str: str) -> Type[Result]:
        """Runs experiment."""
        secret_seq = str_to_byte(secret_str)
        c_oracle = ClassicalOracle(secret=secret_seq)
        q_oracle = QuantumOracle(secret=secret_seq)

        # setup backend
        #self.builder.create_circuit(oracle=q_oracle, random_initialization=True)
        self.builder.create_circuit(oracle=q_oracle, random_initialization=False)
        self.sim.set_noise(config=self.configuration.noise_config)
        self.sim.set_backend(self.configuration.backend)
        self.sim.transpile(circuit=self.builder.circuit, seed=self.configuration.transpiler_seed,
                           config=self.configuration.transpile_config)

        # classical algorithm
        cl_start = perf_counter_ns()
        solution = self.solver.solve(oracle=c_oracle)
        cl_stop = perf_counter_ns()
        cl_time = round((cl_stop - cl_start) / 10 ** 9, 2)

        # quantum algorithm
        qu_start = perf_counter_ns()
        job = self.sim.execute(shots=self.configuration.shot_count,
                               seed_simulator=self.configuration.simulator_seed)
        # noinspection PyUnresolvedReferences
        result = job.result()
        qu_stop = perf_counter_ns()
        qu_time = round((qu_stop - qu_start) / 10 ** 9, 3)

        # construct runtime snapshot
        # create Result data structure
        res = Result
        res.secret = secret_str
        res.result = result

        res.job = job
        res.measurements = result.get_memory()
        res.counts = result.get_counts(self.builder.circuit)
        fill_counts(res.counts, len(secret_str))

        res.cl_result.oracle = c_oracle
        res.cl_result.solution = byte_to_str(solution)
        res.cl_result.time = cl_time

        res.qu_result.oracle = q_oracle
        res.qu_result.solution = dict_max_value_key(res.counts)
        res.qu_result.time = qu_time

        res.snap.configuration = self.configuration
        res.snap.backend_src = self.backend_src
        res.snap.backend_db = self.backend_db
        res.snap.solver = self.solver
        res.snap.builder = self.builder
        res.snap.sim = self.sim

        return res


def preprocess(result: Type[Result]) -> Dict[str, Any]:
    """Preprocess all figures and computationally demanding tasks."""
    ctx: Dict[str, Any] = {}

    # calculate experiment metrics
    gates: Dict[str, List[Any]] = {'instruction': [], 'count': []}
    for instruction, count in result.snap.builder.circuit.count_ops().items():
        gates['instruction'].append(instruction)
        gates['count'].append(count)
    ctx['gates'] = gates
    ctx['timestamp'] = timestamp_str()
    new_build = QuantumCircuitBuild()
    new_build.create_circuit(oracle=result.qu_result.oracle, random_initialization=False)
    ctx['qu_qasm'] = new_build.circuit.qasm(formatted=False)
    ctx['counts_json'] = dumps(result.counts, indent=2, sort_keys=True)
    ctx['memory_csv'] = '\n'.join(result.measurements)

    # draw quantum device maps
    ctx['layout_circuit'] = plot_circuit_layout(result.snap.sim.compiled_circuit,
                                                result.snap.sim.backend)
    ctx['map_gate'] = plot_gate_map(result.snap.sim.backend, label_qubits=True, figsize=(12, 6))
    ctx['map_error'] = plot_error_map(result.snap.sim.backend, figsize=(12, 6), show_title=False)

    # draw circuit visualizations
    ctx['circuit'] = result.snap.builder.circuit.draw(output='mpl', scale=1.1, justify='left',
                                                      initial_state=False, plot_barriers=True,
                                                      idle_wires=True, with_layout=True, fold=-1,
                                                      cregbundle=True)
    ctx['circuit_compiled'] = result.snap.sim.compiled_circuit.draw(output='mpl', scale=1,
                                                                    justify='left', fold=-1,
                                                                    initial_state=False,
                                                                    plot_barriers=True,
                                                                    idle_wires=False,
                                                                    with_layout=False,
                                                                    cregbundle=True)

    preprocess_measurement(ctx, result)

    return ctx


def preprocess_measurement(ctx: Dict[str, Any], result: Type[Result]) -> None:
    """Preprocesses measurement section."""
    # draw measurements chart
    xs1 = np.array([int(i, 2) for i in result.measurements])
    ys1 = np.array(list(range(len(xs1))))
    secret_dec = int(result.secret, 2)
    xs1_secret = xs1 == secret_dec
    xs1_not_secret = xs1 != secret_dec
    ctx['scatter_counts'] = plt.figure(figsize=(15, 10))
    axis = ctx['scatter_counts'].add_subplot(1, 1, 1)
    axis.scatter(xs1[xs1_not_secret], ys1[xs1_not_secret], alpha=0.1, color='#6b6b6b')
    axis.scatter(xs1[xs1_secret], ys1[xs1_secret], alpha=0.1, color='#8210d8')
    axis.set_xticks(np.sort(list(set(xs1))))
    axis.grid(which='both', axis='x', color='grey', linewidth=0.5, alpha=0.4)
    axis.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    axis.set_xlabel('Binary measurement', fontsize=13, labelpad=20)
    axis.set_ylabel('Measurement number', fontsize=13, labelpad=20)
    other_patch = Patch(color='#6b6b6b', label='noise')
    secret_patch = Patch(color='#8210d8', label='target')
    axis.legend(handles=[other_patch, secret_patch], loc='upper right')

    # draw counts chart
    xs2 = np.array(list(result.counts.keys()), dtype=str)
    ys2 = np.array(list(result.counts.values()), dtype=int)
    xs2, ys2 = sort_zipped(xs2, ys2)
    pos2 = find_secret(xs2, result.secret)

    ctx['bar_counts'] = plt.figure(figsize=(15, 10))
    axis = ctx['bar_counts'].add_subplot(1, 1, 1)
    bar_c = axis.bar(xs2, ys2, color='#6b6b6b')
    bar_c[pos2].set_color('#8210d8')
    axis.grid(axis='y', color='grey', linewidth=0.5, alpha=0.4)
    axis.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    axis.set_xlabel('Binary measurement', fontsize=13, labelpad=20)
    axis.set_ylabel('Count', fontsize=13, labelpad=20)
    other_patch = Patch(color='#6b6b6b', label='noise')
    secret_patch = Patch(color='#8210d8', label='target')
    axis.legend(handles=[other_patch, secret_patch], loc='upper right')

    # draw simplified counts bar plot
    ctx['bar_counts_minimal'] = plt.figure(figsize=(8, 4))
    axis = ctx['bar_counts_minimal'].add_subplot(1, 1, 1)
    bar_c = axis.bar(xs2, ys2, color='#6b6b6b')
    bar_c[pos2].set_color('#8210d8')
    axis.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    counts = np.array(list(result.counts.values()))
    axis.axhline(y=(counts.mean() + np.max(counts, initial=0.5)) / 2, color='r', linestyle='-')

    preprocess_error_rate(ctx, result)


def preprocess_error_rate(ctx: Dict[str, Any], result: Type[Result]) -> None:
    """Preprocesses error rate section."""
    correct = result.counts[result.secret]
    incorrect = result.snap.configuration.shot_count - result.counts[result.secret]
    total = result.snap.configuration.shot_count

    # calculate measurement metrics
    err_rate_norm = np.e * incorrect / total / (2 ** len(result.secret))
    ctx['correct_rate'] = f'{correct / total * 100:.2f} %'
    ctx['confidence_ratio'] = 'max' if incorrect == 0 else f'{correct / total / err_rate_norm:.2f}'
    ctx['error_rate_norm'] = f'{err_rate_norm * 100:.2f} %'
    ctx['error_rate_total'] = f'{incorrect / total * 100:.2f} %'

    # draw error rate chart
    ctx['pie_error_rate'], (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    overall_ratios = [incorrect / total, correct / total]
    labels = ['noise', 'target']
    wedges, *_ = ax1.pie(overall_ratios, autopct=lambda pct: pct_to_str(pct, total),
                         startangle=-180 * overall_ratios[0], labels=labels, explode=[0, 0.1],
                         colors=['#6b6b6b', '#8210d8'], textprops={'color': 'w'})
    ax1.legend(wedges, labels, title='Measurements', loc='lower center', bbox_to_anchor=(0, 1))

    ax2.axis('off')
    if incorrect != 0:
        preprocess_bar_of_pie(ax1, ax2, correct, result, wedges)


def preprocess_bar_of_pie(ax1: Any, ax2: Any, correct: int, result: Type[Result],
                          wedges: Any) -> None:
    """Joins pie chart with a bar of wrong qubit count distribution."""
    # pylint: disable-msg=too-many-locals
    # computes bar ratios
    counts = {i: 0 for i in range(1, len(result.secret) + 1)}
    for meas in result.measurements:
        if meas != result.secret:
            counts[diff_letters(meas, result.secret)] += 1
    incorrect_qu = np.array([counts[i] for i in range(1, len(result.secret) + 1)])
    incorrect_ratios = incorrect_qu / sum(incorrect_qu)
    incorrect_labels = [f'{i} qu' for i in range(1, len(result.secret) + 1)]
    bottom = 1
    width = 0.2

    # draw bar of pie
    for j, (height, label) in enumerate(reversed([*zip(incorrect_ratios, incorrect_labels)])):
        if round(height, 2) > 0.02:
            bottom -= height
            bar_c = ax2.bar(0, height, width, bottom=bottom, color='#eb4034', label=label,
                            alpha=0.1 + j / (len(result.secret) + 1))
            ax2.bar_label(bar_c, labels=[f'{height:.0%}'])
    ax2.set_title('Distribution of incorrect qubits', pad=15, loc='right')
    ax2.legend(loc='upper right')
    ax2.set_xlim(- 2.5 * width, 2.5 * width)

    if correct != 0:
        preprocess_connecting_lines(ax1, ax2, incorrect_ratios, wedges, width)


def preprocess_connecting_lines(ax1: Any, ax2: Any, correct_ratios: npt.NDArray[np.float64],
                                wedges: Any, width: float) -> None:
    """Adds connecting lines."""
    # compute connecting lines starting points/angles
    theta1, theta2 = wedges[0].theta1, wedges[0].theta2
    center, radius = wedges[0].center, wedges[0].r
    bar_height = sum(correct_ratios)

    # draw top connecting line
    x_values = radius * np.cos(np.pi / 180 * theta2) + center[0]
    y_values = radius * np.sin(np.pi / 180 * theta2) + center[1]
    con = ConnectionPatch(xyA=(-width / 2, bar_height), coordsA=ax2.transData,
                          xyB=(x_values, y_values), coordsB=ax1.transData)
    con.set_color('#000000')
    con.set_linewidth(0.6)
    ax2.add_artist(con)

    # draw bottom connecting line
    x_values = radius * np.cos(np.pi / 180 * theta1) + center[0]
    y_values = radius * np.sin(np.pi / 180 * theta1) + center[1]
    con = ConnectionPatch(xyA=(-width / 2, 0), coordsA=ax2.transData, xyB=(x_values, y_values),
                          coordsB=ax1.transData)
    con.set_color('#000000')
    ax2.add_artist(con)
    con.set_linewidth(0.6)

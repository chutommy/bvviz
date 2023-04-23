"""Powers the experiments."""
from dataclasses import dataclass
from json import dumps as json_dumps
from time import perf_counter_ns
from typing import List

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch, ConnectionPatch as CPatch
from qiskit import result as q_result
from qiskit.providers import Job
from qiskit.visualization import plot_circuit_layout, plot_gate_map, plot_error_map

from bernstein_vazirani import ClassicalOracle, ClassicalSolver, QuantumOracle, QuantumCircuitBuild
from config import Configuration
from data import BackendDB
from simulation import Simulator, BackendService
from utils import str_to_byte, byte_to_str, generate_seed, fill_counts, sort_zipped, pct_to_str, \
    timestamp_str, find_secret, diff_letters


@dataclass
class Result:
    """Is an output of an experiment. It possesses a snapshot of the experiment."""

    secret: str

    job: Job
    measurements: List[str]
    counts: dict
    result: q_result

    configuration: Configuration
    backend_src: BackendService
    backend_db: BackendDB
    solver: ClassicalSolver
    builder: QuantumCircuitBuild
    sim: Simulator

    cl_oracle: ClassicalOracle
    cl_solution: str
    cl_time: int
    qu_oracle: QuantumOracle
    qu_solution: str
    qu_time: int


class Engine:
    """Represents an instance of an experiment."""

    def __init__(self):
        self.configuration = Configuration()
        self.backend_src = BackendService()
        self.backend_db = BackendDB(self.backend_src.list_backends())
        self.solver = ClassicalSolver()
        self.builder = QuantumCircuitBuild()
        self.sim = Simulator()

    def configure(self, config: dict):
        """Configures experiment."""
        self.configuration.backend = self.backend_db[config["backend_choice"]]
        self.configuration.shot_count = config["shots"]
        self.configuration.simulator_seed = generate_seed()
        self.configuration.transpiler_seed = generate_seed()

        self.configuration.noise_config.reset_rate = config["reset_err"]
        self.configuration.noise_config.measure_rate = config["meas_err"]
        self.configuration.noise_config.single_gate_rate = config["single_err"]
        self.configuration.noise_config.double_gate_rate = config["double_err"]

        self.configuration.transpile_config.layout_method = config["layout"]
        self.configuration.transpile_config.routing_method = config["routing"]
        self.configuration.transpile_config.translation_method = config["translation"]
        self.configuration.transpile_config.optimization_level = config["optimization"]
        self.configuration.transpile_config.approximation_degree = config["approx"]

    def check_secret_size(self, secret: str):
        return len(secret) > self.configuration.backend.num_qubits - 1

    def run(self, secret_str: str):
        """Runs experiment."""
        secret_seq = str_to_byte(secret_str)
        c_oracle = ClassicalOracle(secret=secret_seq)
        q_oracle = QuantumOracle(secret=secret_seq)

        # setup backend
        self.builder.create_circuit(oracle=q_oracle, random_initialization=True)
        self.sim.set_noise(config=self.configuration.noise_config)
        self.sim.set_backend(self.configuration.backend)
        self.sim.transpile(circuit=self.builder.circuit, seed=self.configuration.transpiler_seed,
                           config=self.configuration.transpile_config)

        # classical algorithm
        cl_start = perf_counter_ns()
        solution = self.solver.solve(oracle=c_oracle)
        cl_stop = perf_counter_ns()

        # quantum algorithm
        qu_start = perf_counter_ns()
        job = self.sim.execute(shots=self.configuration.shot_count,
                               seed_simulator=self.configuration.simulator_seed)
        # noinspection PyUnresolvedReferences
        result = job.result()
        qu_stop = perf_counter_ns()

        # construct snapshot
        res = Result
        res.secret = secret_str
        res.result = result

        res.job = job
        res.measurements = result.get_memory()
        res.counts = result.get_counts(self.builder.circuit)
        fill_counts(res.counts, len(secret_str))

        res.cl_oracle = c_oracle
        res.cl_solution = byte_to_str(solution)
        res.cl_time = round((cl_stop - cl_start) / 10 ** 9, 2)

        res.qu_oracle = q_oracle
        res.qu_solution = max(res.counts, key=res.counts.get)
        res.qu_time = round((qu_stop - qu_start) / 10 ** 9, 3)

        res.configuration = self.configuration
        res.backend_src = self.backend_src
        res.backend_db = self.backend_db
        res.solver = self.solver
        res.builder = self.builder
        res.sim = self.sim

        return res


def preprocess(result: Result) -> dict:
    proc = {}

    correct = result.counts[result.secret]
    incorrect = result.configuration.shot_count - result.counts[result.secret]
    total = result.configuration.shot_count
    counts = np.asarray(list(result.counts.values()))

    proc["correct_rate"] = f"{correct / total * 100:.2f} %"
    proc["confidence_level"] = "max likelihood" if incorrect == 0 else f"{correct / incorrect:.2f}"
    proc["error_rate_norm"] = f"{2 * incorrect / total / (2 ** len(result.secret) - 1) * 100:.2f} %"
    proc["error_rate_total"] = f"{incorrect / total * 100:.2f} %"

    proc["timestamp"] = timestamp_str()
    proc["qu_qasm"] = QuantumCircuitBuild() \
        .create_circuit(oracle=result.qu_oracle, random_initialization=False) \
        .circuit.qasm(formatted=False)
    proc["counts_json"] = json_dumps(result.counts, indent=2, sort_keys=True)
    proc["memory_csv"] = '\n'.join(result.measurements)

    gates = {"instruction": [], "count": []}
    for instruction, count in result.builder.circuit.count_ops().items():
        gates["instruction"].append(instruction)
        gates["count"].append(count)
    proc["gates"] = gates
    proc["layout_circuit"] = plot_circuit_layout(result.sim.compiled_circuit, result.sim.backend)
    proc["map_gate"] = plot_gate_map(result.sim.backend, label_qubits=True)
    proc["map_error"] = plot_error_map(result.sim.backend, figsize=(12, 10), show_title=False)
    proc["circuit"] = result.builder.circuit.draw(output="mpl", scale=1.1, justify="left", fold=-1,
                                                  initial_state=False, plot_barriers=True,
                                                  idle_wires=True, with_layout=True, cregbundle=True)
    proc["circuit_compiled"] = result.sim.compiled_circuit.draw(output="mpl", scale=1, justify="left",
                                                                fold=-1,
                                                                initial_state=False, plot_barriers=True,
                                                                idle_wires=False, with_layout=False,
                                                                cregbundle=True)

    ys1 = [int(i, 2) for i in result.measurements]
    xs1 = list(range(len(ys1)))
    proc["bar_counts"] = plt.figure(figsize=(12, 6), dpi=200)
    ax = proc["bar_counts"].add_subplot(1, 1, 1)
    ax.scatter(ys1, xs1, alpha=0.1, color="#8210d8")
    ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax.set_xlabel("Binary measurement", fontsize=13, labelpad=20)
    ax.set_ylabel('Measurement number', fontsize=13, labelpad=20)

    xs2 = np.asarray(list(result.counts.keys()))
    ys2 = np.asarray(list(result.counts.values()))
    xs2, ys2 = sort_zipped(xs2, ys2)
    pos2 = find_secret(xs2, result.secret)
    proc["scatter_counts"] = plt.figure(figsize=(12, 6), dpi=200)
    ax = proc["scatter_counts"].add_subplot(1, 1, 1)
    bar = ax.bar(xs2, ys2, color="#6b6b6b")
    bar[pos2].set_color("#8210d8")
    ax.grid(axis='y', color='grey', linewidth=0.5, alpha=0.4)
    ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax.set_xlabel("Binary measurement", fontsize=13, labelpad=20)
    ax.set_ylabel('Count', fontsize=13, labelpad=20)
    other_patch = Patch(color='#6b6b6b', label='noise')
    secret_patch = Patch(color='#8210d8', label='target')
    ax.legend(handles=[other_patch, secret_patch])

    proc["bar_counts_minimal"] = plt.figure(figsize=(12, 5), dpi=200)
    ax = proc["bar_counts_minimal"].add_subplot(1, 1, 1)
    bar = ax.bar(xs2, ys2, color="#6b6b6b")
    bar[pos2].set_color("#8210d8")
    ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax.axhline(y=(counts.mean() + counts.max(initial=0.5)) / 2, color='r', linestyle='-')

    proc["pie_error_rate"], (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5), dpi=200)
    overall_ratios = [incorrect / total, correct / total]
    labels = ['noise', 'target']
    explode = [0, 0.1]
    angle = -180 * overall_ratios[0]
    wedges, *_ = ax1.pie(overall_ratios, autopct=lambda pct: pct_to_str(pct, total), startangle=angle, labels=labels,
                         explode=explode, colors=["#6b6b6b", "#8210d8"], textprops=dict(color="w"))
    ax1.legend(wedges, labels, title="Measurements", loc="upper left")

    ax2.axis('off')
    if incorrect != 0:
        counts = {i: 0 for i in range(1, len(result.secret) + 1)}
        for meas in result.measurements:
            if meas != result.secret:
                counts[diff_letters(meas, result.secret)] += 1
        correct_ratios = np.asarray([counts[i] for i in range(1, len(result.secret) + 1)])
        correct_ratios = correct_ratios / sum(correct_ratios)
        correct_labels = [f"{i} qu" for i in range(1, len(result.secret) + 1)]
        bottom = 1
        width = .2
        for j, (height, label) in enumerate(reversed([*zip(correct_ratios, correct_labels)])):
            bottom -= height
            bc = ax2.bar(0, height, width, bottom=bottom, color='#eb4034', label=label,
                         alpha=0.1 + j / (len(result.secret) + 1))
            ax2.bar_label(bc, labels=[f"{height:.0%}"], label_type='center')

        ax2.set_title('Number of incorrect qubits', pad=15, loc="right")
        ax2.legend()
        ax2.set_xlim(- 2.5 * width, 2.5 * width)

        if correct != 0:
            theta1, theta2 = wedges[0].theta1, wedges[0].theta2
            center, r = wedges[0].center, wedges[0].r
            bar_height = sum(correct_ratios)

            # draw top connecting line
            x = r * np.cos(np.pi / 180 * theta2) + center[0]
            y = r * np.sin(np.pi / 180 * theta2) + center[1]
            con = CPatch(xyA=(-width / 2, bar_height), coordsA=ax2.transData, xyB=(x, y), coordsB=ax1.transData)
            con.set_color("#000000")
            con.set_linewidth(0.6)
            ax2.add_artist(con)

            # draw bottom connecting line
            x = r * np.cos(np.pi / 180 * theta1) + center[0]
            y = r * np.sin(np.pi / 180 * theta1) + center[1]
            con = CPatch(xyA=(-width / 2, 0), coordsA=ax2.transData, xyB=(x, y), coordsB=ax1.transData)
            con.set_color("#000000")
            ax2.add_artist(con)
            con.set_linewidth(0.6)

    return proc

"""Powers the experiments."""
import json
from dataclasses import dataclass
from time import perf_counter_ns
from typing import List

import pandas as pd
import streamlit as st
from qiskit.providers import Job
from qiskit.visualization import plot_circuit_layout, plot_gate_map, plot_error_map

from bernstein_vazirani import ClassicalOracle, ClassicalSolver, QuantumOracle, QuantumCircuitBuild
from config import LayoutMethod, RoutingMethod, TranslationMethod, Configuration, OptimizationLevel
from data import BackendDB, Descriptor
from simulation import Simulator, BackendService
from utils import str_to_byte, timestamp_str, byte_to_str, method_to_name, optimization_to_name, \
    backend_to_name, generate_seed


@dataclass
class Result:
    """Is an output of an experiment. It possesses a snapshot of the experiment."""

    job: Job
    measurements: List[str]
    counts: dict

    cl_oracle: ClassicalOracle
    cl_solution: str
    cl_time: int

    qu_oracle: QuantumOracle
    qu_solution: str
    qu_time: int


class Engine:
    """Represents an instance of an experiment."""

    configuration: Configuration = None
    backed: BackendService = None
    backend_db: BackendDB = None

    def __init__(self):
        self.configuration = Configuration()
        self.backed = BackendService()
        self.backend_db = BackendDB(self.backed.list_backends())
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

        # construct result
        res = Result

        res.job = job
        res.measurements = result.get_memory()
        res.counts = result.get_counts(self.builder.circuit)

        res.cl_oracle = c_oracle
        res.cl_solution = byte_to_str(solution)
        res.cl_time = round((cl_stop - cl_start) / 10 ** 9, 2)

        res.qu_oracle = q_oracle
        res.qu_solution = max(res.counts, key=res.counts.get)
        res.qu_time = round((qu_stop - qu_start) / 10 ** 9, 3)

        return res

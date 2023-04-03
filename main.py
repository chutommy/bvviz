from time import perf_counter_ns

import matplotlib.pyplot as plt
from qiskit.visualization import plot_circuit_layout, plot_gate_map, plot_error_map

from bernstein_vazirani import ClassicalOracle, ClassicalSolver, QuantumOracle, QuantumCircuitBuild
from config import Backend, LayoutMethod, RoutingMethod, TranslationMethod, Configuration, \
    OptimizationLevel
from simulation import Simulator
from utils import str_to_byte

# ======================================

cfg = Configuration()

cfg.seed = 1406711823  # randint(10 ** 9, 10 ** 10)
cfg.backend = Backend.MELBOURNE
cfg.shot_count = 1000

cfg.noise_config.reset_rate = 0.01
cfg.noise_config.measure_rate = 0.05
cfg.noise_config.single_gate_rate = 0.07
cfg.noise_config.double_gate_rate = 0.11

cfg.transpile_config.layout_method = LayoutMethod.NOISE_ADAPTIVE
cfg.transpile_config.routing_method = RoutingMethod.LOOKAHEAD
cfg.transpile_config.translation_method = TranslationMethod.SYNTHESIS

cfg.transpile_config.approximation_degree = 0.99
cfg.transpile_config.optimization_level = OptimizationLevel.HEAVY

secret_str = "10100111011"

# ======================================

secret_seq = str_to_byte(secret_str)

solver = ClassicalSolver()
builder = QuantumCircuitBuild()

oracle = ClassicalOracle(secret=secret_seq)
q_oracle = QuantumOracle(secret=secret_seq)

builder.create_circuit(oracle=q_oracle, random_initialization=True)
sim = Simulator()
sim.set_noise(config=cfg.noise_config)
sim.set_backend(cfg.backend.value)
sim.transpile(circuit=builder.circuit, seed=cfg.seed, config=cfg.transpile_config)

cl_start = perf_counter_ns()
solution = solver.solve(oracle=oracle)
cl_stop = perf_counter_ns()

qu_start = perf_counter_ns()
job = sim.execute(shots=cfg.shot_count, seed_simulator=cfg.seed)
qu_stop = perf_counter_ns()

# ======================================

# noinspection PyUnresolvedReferences
result = job.result()
counts = result.get_counts(builder.circuit)

cl_solution = solution
cl_queries = oracle.query_count
cl_bytecode_instructions = solver.ops_count()
cl_time = cl_start - cl_stop

qu_queries = q_oracle.query_count
qu_used_gates = builder.circuit.count_ops()
qu_gates_count = builder.circuit.size()
qu_global_phase = builder.circuit.global_phase
qu_qubit_count = builder.circuit.num_qubits
qu_clbit_count = builder.circuit.num_clbits
qu_time = qu_start - qu_stop
qu_qasm = QuantumCircuitBuild().create_circuit(
    oracle=q_oracle, random_initialization=False).circuit.qasm(formatted=False)

# noinspection PyUnresolvedReferences
be_backend_name = job.backend()
be_version = sim.backend.backend_version
be_qubit_capacity = sim.backend.num_qubits
# noinspection PyUnresolvedReferences
job_id = job.job_id()
# noinspection PyUnresolvedReferences
job_success = job.result().success

# ======================================

sim.compiled_circuit.draw(output="mpl")
plt.show()
plt.close()

plot_circuit_layout(sim.compiled_circuit, sim.backend)
plt.show()
plt.close()

plot_gate_map(sim.backend)
plt.show()
plt.close()

plot_error_map(sim.backend)
plt.show()
plt.close()

yss = result.get_memory()
ys = [int(i, 2) for i in yss]
xs = list(range(len(ys)))
plt.scatter(xs, ys, alpha=0.1)
plt.tick_params(
    axis='y',
    which='both',
    labelleft=False)
plt.show()
plt.close()

builder.circuit.draw(output="mpl",
                     initial_state=True,
                     plot_barriers=False,
                     interactive=True,
                     fold=-1)
plt.show()
plt.close()

courses = list(counts.keys())
values = list(counts.values())
plt.bar(courses, values)
plt.tick_params(
    axis='x',
    which='both',
    bottom=False,
    top=False,
    labelbottom=False)
plt.show()
plt.close()

from random import randint
from time import perf_counter_ns

import matplotlib.pyplot as plt
from qiskit.providers.fake_provider import FakeGuadalupeV2
from qiskit.visualization import plot_circuit_layout, plot_gate_map, plot_error_map

from backend_simulator import Simulator
from bernstein_vazirani import ClassicalOracle, ClassicalSolver, QuantumOracle, QuantumCircuitBuild
from utils import str_to_byte

# ======================================

SECRET_STR = "101001110"

RESET_RATE = 0.01
MEASURE_RATE = 0.05
SINGLE_GATE_RATE = 0.07
DOUBLE_GATE_RATE = 0.11

LAYOUT_METHOD = "sabre"
ROUTING_METHOD = "stochastic"
TRANSLATION_METHOD = "synthesis"
APPROXIMATION_DEGREE = 0
OPTIMIZATION_LEVEL = 2

SHOT_COUNT = 1000

backend = FakeGuadalupeV2()
random_seed = randint(10 ** 9, 10 ** 10)

# ======================================

solver = ClassicalSolver()
secret_seq = str_to_byte(SECRET_STR)
oracle = ClassicalOracle(secret=secret_seq)
cl_start = perf_counter_ns()
solution = solver.solve(oracle=oracle)
cl_stop = perf_counter_ns()

builder = QuantumCircuitBuild()
q_oracle = QuantumOracle(secret=secret_seq)
builder.create_circuit(oracle=q_oracle, random_initialization=True)
sim = Simulator()
sim.set_noise(reset_rate=RESET_RATE,
              measure_rate=MEASURE_RATE,
              single_gate_rate=SINGLE_GATE_RATE,
              double_gate_rate=DOUBLE_GATE_RATE)
sim.set_backend(backend)
sim.transpile(circuit=builder.circuit,
              seed_transpiler=random_seed,
              layout_method=LAYOUT_METHOD,
              routing_method=ROUTING_METHOD,
              translation_method=TRANSLATION_METHOD,
              approximation_degree=APPROXIMATION_DEGREE,
              optimization_level=OPTIMIZATION_LEVEL)
qu_start = perf_counter_ns()
job = sim.execute(shots=SHOT_COUNT, seed_simulator=random_seed)
qu_stop = perf_counter_ns()
# noinspection PyUnresolvedReferences
result = job.result()
counts = result.get_counts(builder.circuit)

# ======================================

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
                     interactive=True)
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

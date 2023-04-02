from random import randint

import matplotlib.pyplot as plt
from qiskit.providers.fake_provider import FakeGuadalupeV2
from qiskit.visualization import plot_circuit_layout, plot_gate_map, plot_error_map

from backend_simulator import *
from bernstein_vazirani import *

secret = np.array([1, 0, 1, 0, 0, 1, 1, 1, 0], dtype=np.byte)

oracle = ClassicalOracle(secret=secret)
solver = ClassicalSolver()
solution = solver.solve(oracle=oracle)

print(solution)

print("secret string:".ljust(20, " "), secret)
print("classical solution:".ljust(20, " "), solution)
print("match:".ljust(20, " "), np.array_equal(solution, secret))
print("# of queries:".ljust(20, " "), oracle.query_count)

# =============================================

q_oracle = QuantumOracle(secret=secret)
builder = QuantumCircuitBuild()
builder.create_circuit(oracle=q_oracle)
qc = builder.circuit

print("classical ops", solver.ops_count())
print("qc ops", builder.circuit.count_ops())
print("qc size", builder.circuit.size())
print("qc global phase", builder.circuit.global_phase)
print("qc qubits", builder.circuit.num_qubits)
print("qc cbits", builder.circuit.num_clbits)

# =============================================

sim = Simulator()
sim.set_noise(reset_rate=0.01, measure_rate=0.05, single_gate_rate=0.07, double_gate_rate=0.11)
basis_gates = sim.noise_config.model.basis_gates

random_seed = randint(1000, 1000 ** 3)

sim.set_backend(FakeGuadalupeV2())
qc_compiled = sim.transpile(circuit=builder.circuit, layout_method="sabre", routing_method="stochastic",
                            translation_method="synthesis", approximation_degree=0,
                            seed_transpiler=random_seed, optimization_level=2)
job = sim.execute(compiled_circuit=qc_compiled, shots=1000, seed_simulator=random_seed)
result = job.result()
counts = result.get_counts(qc)

# =========================

qc_compiled.draw(output="mpl")
plt.show()
plt.close()

plot_circuit_layout(qc_compiled, sim.backend)
plt.show()
plt.close()

plot_gate_map(sim.backend)
plt.show()
plt.close()

plot_error_map(sim.backend)
plt.show()
plt.close()

ys = result.get_memory()
xs = [x for x in range(len(ys))]
plt.scatter(xs, ys)
plt.tick_params(
    axis='y',
    which='both',
    labelleft=False)
plt.show()
plt.close()

qc.draw(output="mpl", initial_state=True, plot_barriers=False, interactive=True)
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

print("==========================")

print(sim.backend.num_qubits)
print(sim.backend.backend_version)
print(sim.backend.operation_names)
print(sim.backend.version)
print(job.backend())
print(job.job_id())
print(job.result().success)

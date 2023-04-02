from random import randint

import matplotlib.pyplot as plt
from qiskit.providers.fake_provider import FakeGuadalupeV2
from qiskit.visualization import plot_circuit_layout

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

qoracle = QuantumOracle(secret=secret)
builder = QuantumCircuitBuild()
builder.create_circuit(oracle=qoracle)
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

sim.set_backend(FakeGuadalupeV2())
qc_compiled = sim.transpile(builder.circuit, "sabre", "stochastic", "synthesis", 0, randint(1000, 1000 ** 3), 2)
job = sim.backend.run(qc_compiled, shots=800, basis_gates=basis_gates, noise_model=sim.noise_config.model)
result = job.result()
counts = result.get_counts(qc)

# creating the dataset
courses = list(counts.keys())
values = list(counts.values())

# =========================

qc_compiled.draw(output="mpl")
plt.show()

plot_circuit_layout(qc_compiled, sim.backend)
plt.show()

qc.draw(output="mpl", initial_state=True, plot_barriers=False, interactive=True)
plt.show()

plt.bar(courses, values)
plt.tick_params(
    axis='x',  # changes apply to the x-axis
    which='both',  # both major and minor ticks are affected
    bottom=False,  # ticks along the bottom edge are off
    top=False,  # ticks along the top edge are off
    labelbottom=False)  # labels along the bottom edge are off
plt.show()

# config = backend.configuration()
# props = backend.properties()
# print()

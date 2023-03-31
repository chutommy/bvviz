from qiskit import Aer, transpile

from builder import *

secret = np.array([1, 0, 1, 0, 0, 1, 1, 1, 0], dtype=np.byte)

oracle = ClassicalOracle(secret)
solver = ClassicalSolver(oracle)
solution = solver.solve()

print(solution)

print("secret string:".ljust(20, " "), secret)
print("classical solution:".ljust(20, " "), solution)
print("match:".ljust(20, " "), np.array_equal(solution, secret))
print("# of queries:".ljust(20, " "), oracle.query_count)

# =============================================

qoracle = QuantumOracle(secret)
builder = QuantumCircuitBuild(qoracle)
builder.create_circuit()
qc = builder.circuit
qc.draw(output="mpl", initial_state=True, plot_barriers=False, interactive=True)

# =============================================

noise_build = NoiseBuild()

noise_build.applyResetError()
noise_build.applyMeasurementError()
noise_build.applyGateError()

basis_gates = noise_build.model.basis_gates

backend = Aer.get_backend("qasm_simulator")

qc_compiled = transpile(builder.circuit, backend)
job = backend.run(qc_compiled, shots=800, basis_gates=basis_gates, noise_model=noise_build.model)
result = job.result()
counts = result.get_counts(qc)

import matplotlib.pyplot as plt

# creating the dataset
courses = list(counts.keys())
values = list(counts.values())

plt.bar(courses, values)

plt.tick_params(
    axis='x',  # changes apply to the x-axis
    which='both',  # both major and minor ticks are affected
    bottom=False,  # ticks along the bottom edge are off
    top=False,  # ticks along the top edge are off
    labelbottom=False)  # labels along the bottom edge are off

plt.show()
config = backend.configuration()
print()

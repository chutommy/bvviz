from random import randint
from time import perf_counter_ns

import streamlit as st
from qiskit.providers import Backend

from bernstein_vazirani import ClassicalOracle, ClassicalSolver, QuantumOracle, QuantumCircuitBuild
from config import LayoutMethod, RoutingMethod, TranslationMethod, Configuration, OptimizationLevel
from simulation import Simulator, BackendService
from utils import str_to_byte


# ======================================
def backend_to_name(backend: Backend) -> str:
    """Extract the name and number of qubits from the provider's fake backend system identifier."""
    # fake_######_v2
    name = backend.name
    if name.startswith("fake_"):
        name = name[5:]
    if name.endswith("_v2"):
        name = name[:-3]
        name = name.capitalize().replace("_", " ")
    return f"{name} ({backend.num_qubits})"


def method_to_name(method: str) -> str:
    """Returns a formatted name of the method."""
    # fake_######_v2
    return method.replace("_", " ").capitalize()


def optimization_to_name(level: int) -> str:
    """Map the enum title to the value."""
    return OptimizationLevel(level).name.replace("_", " ").capitalize()


# ======================================

st.title("Bernstein–Vazirani problem")

cfg = Configuration()
be = BackendService()

if 'init' not in st.session_state:
    st.session_state.init = True
    st.session_state.transpiler_seed = randint(10 ** 9, 10 ** 10)
    st.session_state.simulator_seed = randint(10 ** 9, 10 ** 10)
    st.session_state.shots = 1000

    st.session_state.reset_rate = 0.001
    st.session_state.measure_rate = 0.032
    st.session_state.single_gate_rate = 0.051
    st.session_state.double_gate_rate = 0.073

with st.sidebar.form("configuration", clear_on_submit=False):
    st.header("Configuration")

    st.subheader("Backend")

    cfg.backend = st.selectbox("Quantum system", options=be.list_backends(), index=0, format_func=backend_to_name,
                               help="Choose a quantum simulator backend for the experiment. "
                                    "The number next to each backend name indicates the maximum number of qubits the simulator can handle.")

    cfg.shot_count = st.number_input("Shots", min_value=1, max_value=10 ** 5,
                                     value=st.session_state.shots, step=1,
                                     help="Enter the number of times the circuit will be executed, providing statistical results from multiple measurements. "
                                          "Consider a higher number of shots for better accuracy, but note that it will also increase the computational time.")

    cfg.simulator_seed = st.number_input("Simulator seed", min_value=0, max_value=10 ** 15,
                                         value=st.session_state.simulator_seed, step=1,
                                         help="Seed to control simulator sampling.",
                                         label_visibility="visible")

    st.divider()

    st.subheader("Noise model")

    cfg.noise_config.reset_rate = st.slider("Reset Error Rate", min_value=0.0, max_value=0.01,
                                            value=st.session_state.reset_rate, step=0.0001, format="%.4f",
                                            help="Specify the error rate for qubit reset operation, which is the probability that a qubit fails to be reset to the initial state.", )

    cfg.noise_config.measure_rate = st.slider("Measure Error Rate", min_value=0.0, max_value=0.2,
                                              value=st.session_state.measure_rate, step=0.001, format="%.3f",
                                              help="Specify the error rate for the measurement operation, which is the probability of obtaining an incorrect outcome after performing a measurement on a single qubit.", )

    cfg.noise_config.single_gate_rate = st.slider("Single Gate Error Rate", min_value=0.0, max_value=0.4,
                                                  value=st.session_state.single_gate_rate, step=0.001, format="%.3f",
                                                  help="Specify the error rate for single-qubit gates, which models the probability of error during the execution of a single-qubit gate operations (X, H).", )

    cfg.noise_config.double_gate_rate = st.slider("Two Gate Error Rate", min_value=0.0, max_value=0.6,
                                                  value=st.session_state.double_gate_rate, step=0.001, format="%.3f",
                                                  help="Specify the error rate for two-qubit gates, which models the probability of error during the execution of a two-qubit gate operation (CNOT).", )

    st.divider()

    st.subheader("Transpiler")

    cfg.transpile_config.layout_method = st.selectbox("Layout Method", options=[lm.value for lm in LayoutMethod],
                                                      index=2, format_func=method_to_name,
                                                      help="Choose a layout method for the transpiler to map the circuit qubits to physical qubits on the quantum hardware.")

    cfg.transpile_config.routing_method = st.selectbox("Routing Method", options=[rm.value for rm in RoutingMethod],
                                                       index=1, format_func=method_to_name,
                                                       help="Choose a layout method for the transpiler to map the circuit qubits to physical qubits on the quantum hardware.")

    cfg.transpile_config.translation_method = st.selectbox("Translation Method",
                                                           options=[tm.value for tm in TranslationMethod],
                                                           index=1, format_func=method_to_name,
                                                           help="Choose a layout method for the transpiler to map the circuit qubits to physical qubits on the quantum hardware.")

    cfg.transpile_config.optimization_level = st.select_slider("Optimization Level",
                                                               options=[ol.value for ol in OptimizationLevel],
                                                               value=1, format_func=optimization_to_name,
                                                               help="Choose a layout method for the transpiler to map the circuit qubits to physical qubits on the quantum hardware.")

    cfg.transpiler_seed = st.number_input("Transpiler seed", min_value=0, max_value=10 ** 15,
                                          value=st.session_state.transpiler_seed, step=1,
                                          help="Seed for the stochastic parts of the transpiler.",
                                          label_visibility="visible")

    submitted = st.form_submit_button("Execute", type="primary", disabled=False, use_container_width=True)

cfg.transpile_config.approximation_degree = 0.99

secret_str = "1011"

# ======================================

secret_seq = str_to_byte(secret_str)

solver = ClassicalSolver()
builder = QuantumCircuitBuild()

oracle = ClassicalOracle(secret=secret_seq)
q_oracle = QuantumOracle(secret=secret_seq)

builder.create_circuit(oracle=q_oracle, random_initialization=True)
sim = Simulator()
sim.set_noise(config=cfg.noise_config)
sim.set_backend(cfg.backend)
sim.transpile(circuit=builder.circuit, seed=cfg.transpiler_seed, config=cfg.transpile_config)

cl_start = perf_counter_ns()
solution = solver.solve(oracle=oracle)
cl_stop = perf_counter_ns()

qu_start = perf_counter_ns()
job = sim.execute(shots=cfg.shot_count, seed_simulator=cfg.simulator_seed)
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
qu_qasm = QuantumCircuitBuild() \
    .create_circuit(oracle=q_oracle, random_initialization=False) \
    .circuit.qasm(formatted=False)

# noinspection PyUnresolvedReferences
be_backend_name = job.backend()
be_version = sim.backend.backend_version
be_qubit_capacity = sim.backend.num_qubits
# noinspection PyUnresolvedReferences
job_id = job.job_id()
# noinspection PyUnresolvedReferences
job_success = job.result().success

# ======================================

# sim.compiled_circuit.draw(output="mpl")
# plt.show()
# plt.close()
#
# plot_circuit_layout(sim.compiled_circuit, sim.backend)
# plt.show()
# plt.close()
#
# plot_gate_map(sim.backend)
# plt.show()
# plt.close()
#
# plot_error_map(sim.backend)
# plt.show()
# plt.close()
#
# yss = result.get_memory()
# ys = [int(i, 2) for i in yss]
# xs = list(range(len(ys)))
# plt.scatter(xs, ys, alpha=0.1)
# plt.tick_params(
#     axis='y',
#     which='both',
#     labelleft=False)
# plt.show()
# plt.close()
#
# builder.circuit.draw(output="mpl",
#                      initial_state=True,
#                      plot_barriers=False,
#                      interactive=True,
#                      fold=-1)
# plt.show()
# plt.close()
#
# courses = list(counts.keys())
# values = list(counts.values())
# plt.bar(courses, values)
# plt.tick_params(
#     axis='x',
#     which='both',
#     bottom=False,
#     top=False,
#     labelbottom=False)
# plt.show()
# plt.close()

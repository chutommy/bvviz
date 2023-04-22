import json
from random import randint
from time import perf_counter_ns

import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt

from bernstein_vazirani import ClassicalOracle, ClassicalSolver, QuantumOracle, QuantumCircuitBuild
from config import LayoutMethod, RoutingMethod, TranslationMethod, Configuration, OptimizationLevel
from simulation import Simulator, BackendService
from utils import str_to_byte, timestamp_str, byte_to_str, method_to_name, \
    optimization_to_name, backend_to_name

# ======================================

st.set_page_config(page_title="Bernstein–Vazirani Algorithm", page_icon="assets/logo.png",
                   layout="centered", initial_sidebar_state="auto", menu_items=None)
st.markdown("<style>header{visibility: hidden;}</style>", unsafe_allow_html=True)
st.title("Bernstein–Vazirani", anchor=False)

cfg = Configuration()
be = BackendService()
be_list = list(be.list_backends())

if 'init' not in st.session_state:
    st.session_state.init = True

    st.session_state.transpiler_seed = randint(10 ** 9, 10 ** 10)
    st.session_state.simulator_seed = randint(10 ** 9, 10 ** 10)
    st.session_state.shots = 1000

    st.session_state.reset_rate = 0.006
    st.session_state.measure_rate = 0.042
    st.session_state.single_gate_rate = 0.061
    st.session_state.double_gate_rate = 0.077

    st.session_state.approximation_degree = 0.99

failed = False
with st.sidebar.form("configuration", clear_on_submit=False):
    st.header("Configuration")

    st.subheader("Backend")

    backend_choice = st.selectbox("Quantum system", options=range(len(be_list)), index=0,
                                  format_func=lambda id: backend_to_name(be_list[id]),
                                  help="Choose a quantum simulator backend for the experiment. The number next to each backend name indicates the maximum number of qubits the simulator can handle.")
    cfg.backend = be_list[backend_choice]

    cfg.shot_count = st.number_input("Shots", min_value=1, max_value=10 ** 5,
                                     value=st.session_state.shots, step=1,
                                     help="Enter the number of times the circuit will be executed, providing statistical results from multiple measurements. Consider a higher number of shots for better accuracy, but note that it will also increase the computational time.")

    # cfg.simulator_seed = st.number_input("Simulator seed", min_value=0, max_value=10 ** 15,
    #                                      value=st.session_state.simulator_seed, step=1,
    #                                      help="Seed to control simulator sampling.",
    #                                      label_visibility="visible")
    cfg.simulator_seed = randint(10 ** 9, 10 ** 10)

    st.divider()

    st.subheader("Input")

    secret_str = st.text_input("Secret string", value="1101",
                               help="Enter a secret string for the Bernstein-Vazirani algorithm, which determines the value to be discovered using the quantum circuit. This string must consist of 0s and 1s. Note that the length of the secret string must not exceed the number of qubits in the backend simulator minus one, as one qubit is reserved for the ancilla qubit.")
    if len(secret_str) > cfg.backend.num_qubits - 1:
        st.error(
            f"The length of the secret string ({len(secret_str)}) exceeds the number of qubits in the backend simulator minus one ({cfg.backend.num_qubits} - 1).")
        failed = True
    elif not all(c in '01' for c in secret_str):
        st.error("The secret string should only consist of 0s and 1s.")
        failed = True

    st.divider()

    st.subheader("Noise model")

    cfg.noise_config.reset_rate = st.slider("Reset error rate", min_value=0.0, max_value=0.1,
                                            value=st.session_state.reset_rate, step=0.001,
                                            format="%.4f",
                                            help="Specify the error rate for qubit reset operation, which is the probability that a qubit fails to be reset to the initial state.", )

    cfg.noise_config.measure_rate = st.slider("Measure error rate", min_value=0.0, max_value=0.5,
                                              value=st.session_state.measure_rate, step=0.001,
                                              format="%.3f",
                                              help="Specify the error rate for the measurement operation, which is the probability of obtaining an incorrect outcome after performing a measurement on a single qubit.", )

    cfg.noise_config.single_gate_rate = st.slider("Single Gate error rate",
                                                  min_value=0.0, max_value=0.5,
                                                  value=st.session_state.single_gate_rate,
                                                  step=0.001, format="%.3f",
                                                  help="Specify the error rate for single-qubit gates, which models the probability of error during the execution of a single-qubit gate operations (X, H).", )

    cfg.noise_config.double_gate_rate = st.slider("Two Gate error rate",
                                                  min_value=0.0, max_value=0.5,
                                                  value=st.session_state.double_gate_rate,
                                                  step=0.001, format="%.3f",
                                                  help="Specify the error rate for two-qubit gates, which models the probability of error during the execution of a two-qubit gate operation (CNOT).", )

    st.divider()

    st.subheader("Transpiler")

    cfg.transpile_config.layout_method = st.selectbox("Layout method",
                                                      options=[lm.value for lm in LayoutMethod],
                                                      index=2, format_func=method_to_name,
                                                      help="Choose a layout method for the transpiler to map the circuit qubits to physical qubits on the quantum hardware.")

    cfg.transpile_config.routing_method = st.selectbox("Routing method",
                                                       options=[rm.value for rm in RoutingMethod],
                                                       index=1, format_func=method_to_name,
                                                       help="Choose a routing method for the transpiler to optimize the qubit connections and minimize the errors introduced during the circuit execution.")

    cfg.transpile_config.translation_method = st.selectbox("Translation method",
                                                           options=[tm.value for tm in
                                                                    TranslationMethod],
                                                           index=1, format_func=method_to_name,
                                                           help="Choose a translation method for the transpiler to convert the circuit instructions into the instructions compatible with the selected backend.")

    cfg.transpile_config.optimization_level = st.select_slider("Optimization level",
                                                               options=[ol.value for ol in
                                                                        OptimizationLevel],
                                                               value=1,
                                                               format_func=optimization_to_name,
                                                               help="Select an optimization level for the transpiler to optimize the circuit's performance by reducing the number of gates, reducing the circuit depth, or minimizing the number of SWAP gates required for qubit mapping. The higher the optimization level, the more aggressive the optimization process, which can lead to faster execution time but may also affect the circuit's accuracy.")

    cfg.transpile_config.approximation_degree = st.slider("Approximation degree", min_value=0.9,
                                                          max_value=1.0,
                                                          value=st.session_state.approximation_degree,
                                                          step=0.01,
                                                          format="%.2f",
                                                          help="Specify an approximation degree for the transpiler to approximate the circuit's gates using a lower number of gates, reducing the circuit's overall complexity.", )

    # cfg.transpiler_seed = st.number_input("Transpiler seed", min_value=0, max_value=10 ** 15,
    #                                       value=st.session_state.transpiler_seed, step=1,
    #                                       help="Seed for the stochastic parts of the transpiler.")
    cfg.transpiler_seed = randint(10 ** 9, 10 ** 10)

    submitted = st.form_submit_button("Execute", type="primary", disabled=False,
                                      use_container_width=True)

if failed:
    st.warning(
        "Execution failed due to invalid configuration settings. Please ensure all values are valid and fully compatible with the selected backend before taking the next step.")
    st.stop()

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
cl_solution = byte_to_str(solution)
cl_time = round((cl_stop - cl_start) / 10 ** 9, 2)

qu_start = perf_counter_ns()
job = sim.execute(shots=cfg.shot_count, seed_simulator=cfg.simulator_seed)
result = job.result()
measurements = result.get_memory()
counts = result.get_counts(builder.circuit)
qu_stop = perf_counter_ns()
qu_time = round((qu_stop - qu_start) / 10 ** 9, 3)

qu_solution = max(counts, key=counts.get)
qu_solution_ok = qu_solution == secret_str

cl_queries = oracle.query_count
qu_queries = q_oracle.query_count

# ======================================

solution_cols = st.columns(2)
# solution_cols[0].metric("Secret string", value=secret_str)
with solution_cols[0]:
    st.caption(":orange[Classical] approach")
    st.metric(":orange[CL] solution", value=cl_solution, delta="OK",
              help="The solution obtained by the classical algorithm which was computed using classical computation methods.")

    oracle_cols = st.columns(2)
    oracle_cols[0].metric(":orange[CL] duration", value=f"{cl_time}s")
    oracle_cols[1].metric(":orange[CL] queries count", value=f"{cl_queries}x")

with solution_cols[1]:
    st.caption(":violet[Quantum] approach")
    st.metric(":violet[QU] solution", value=qu_solution,
              delta="OK" if qu_solution_ok else "BAD",
              delta_color="normal" if qu_solution_ok else "inverse",
              help="The solution obtained by the quantum circuit which was computed using quantum computation methods")

    oracle_cols = st.columns(2)
    oracle_cols[0].metric(":violet[QU] duration", value=f"{qu_time}s")
    oracle_cols[1].metric(":violet[QU] queries count", value=f"{qu_queries}x")

st.divider()

backend_cols = st.columns(2)

with backend_cols[0]:
    st.subheader("Quantum hardware", anchor=False)

    qu_clbit_count = builder.circuit.num_clbits
    qu_gates_count = builder.circuit.size()
    qu_qubit_count = builder.circuit.num_qubits
    qu_global_phase = builder.circuit.global_phase
    be_qubit_capacity = sim.backend.num_qubits

    be_backend_name = job.backend()
    be_version = sim.backend.backend_version
    job_id = job.job_id()
    job_success = job.result().success

    metric_cols = st.columns(2)
    metric_cols2 = st.columns(2)
    metric_cols[0].metric("Classical bits", value=f"{qu_clbit_count}b")
    metric_cols[1].metric("Quantum bits", value=f"{qu_qubit_count}qu")
    metric_cols2[0].metric("Quantum gates", value=f"{qu_gates_count}")
    metric_cols2[1].metric("Quantum bits (cap)", value=f"{be_qubit_capacity}qu")
    # metric_cols2[1].metric("Global phase", value=f"{round(qu_global_phase, 2)}π")

    if job_success:
        st.success(f"{be_backend_name} {be_version} (:green[success])")
    else:
        st.error(f"{be_backend_name} {be_version} (':red[fail]')")

with backend_cols[1]:
    qu_used_gates = builder.circuit.count_ops()
    gates = {"instruction": [], "count": []}
    for instruction, count in qu_used_gates.items():
        gates["instruction"].append(instruction)
        gates["count"].append(count)
    df = pd.DataFrame.from_dict(gates)
    st.dataframe(df, use_container_width=True)

st.divider()

download_cols = st.columns(4)
timestamp = timestamp_str()

download_cols[0].subheader("Downloads:", anchor=False)

qu_qasm = QuantumCircuitBuild() \
    .create_circuit(oracle=q_oracle, random_initialization=False) \
    .circuit.qasm(formatted=False)
download_cols[1].download_button("OpenQASM (qasm)", data=qu_qasm, mime="text/plain",
                                 help="Download the OpenQASM code for the circuit used in the experiment.",
                                 file_name=f"bernstein_vazirani_{timestamp}.qasm",
                                 use_container_width=True)

memory_csv = '\n'.join(measurements)
download_cols[2].download_button("Measurements (CSV)", data=memory_csv, mime="text/csv",
                                 help="Download measurements of the experiment as a CSV file. The file will contain raw data from the experiment, including the binary outcome of each measurement in the order in which they were taken. The file is saved in a comma-separated value format and can be imported into spreadsheet or analysis software for further processing or visualization.",
                                 file_name=f"bernstein_vazirani_{timestamp}.csv",
                                 use_container_width=True)

counts_json = json.dumps(counts, indent=2, sort_keys=True)
download_cols[3].download_button("Counts (JSON)", data=counts_json, mime="application/json",
                                 help="Download the counts of the experiment as a JSON file. The file will contain raw data, including the counts of each measured state. Consider using this data for further analysis or visualization.",
                                 file_name=f"bernstein_vazirani_{timestamp}.json",
                                 use_container_width=True)

# ======================================

# fig = sim.compiled_circuit.draw(output="mpl")
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

yss = measurements
ys = [int(i, 2) for i in yss]
xs = list(range(len(ys)))

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(xs, ys, alpha=0.1)
ax.tick_params(
    axis='y',
    which='both',
    labelleft=False)
st.pyplot(fig)

# builder.circuit.draw(output="mpl",
#                      initial_state=True,
#                      plot_barriers=False,
#                      interactive=True,
#                      fold=-1)
# plt.show()
# plt.close()

courses = list(counts.keys())
values = list(counts.values())

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.bar(courses, values)
ax.tick_params(
    axis='x',
    which='both',
    bottom=False,
    top=False,
    labelbottom=False)
st.pyplot(fig)

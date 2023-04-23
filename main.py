import json
from time import perf_counter_ns

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
from matplotlib.patches import Patch, ConnectionPatch
from qiskit.visualization import plot_circuit_layout, plot_gate_map, plot_error_map
from scipy.interpolate import make_interp_spline, interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gaussian_kde

from bernstein_vazirani import ClassicalOracle, ClassicalSolver, QuantumOracle, QuantumCircuitBuild
from config import LayoutMethod, RoutingMethod, TranslationMethod, Configuration, OptimizationLevel
from data import BackendDB, Descriptor
from engine import Engine
from simulation import Simulator, BackendService
from utils import str_to_byte, timestamp_str, byte_to_str, method_to_name, optimization_to_name, \
    backend_to_name, generate_seed, all_measurement_outputs, sort_zipped, diff_letters

descriptor = Descriptor('assets/descriptions.json')
engine = Engine()

# ====================================================================

st.set_page_config(page_title="Bernstein–Vazirani", page_icon="assets/logo.png",
                   layout="wide", initial_sidebar_state="auto", menu_items=None)
st.markdown(descriptor.cat(["style_hide_header", "style_hide_footer",
                            "style_hide_view_fullscreen"]), unsafe_allow_html=True)
st.title("Bernstein–Vazirani Quantum Protocol", anchor=False)
st.divider()

if 'init' not in st.session_state:
    st.session_state.init = True

    st.session_state.transpiler_seed = generate_seed()
    st.session_state.simulator_seed = generate_seed()
    st.session_state.shots = 1000

    st.session_state.reset_rate = 0.006
    st.session_state.measure_rate = 0.042
    st.session_state.single_gate_rate = 0.061
    st.session_state.double_gate_rate = 0.077

    st.session_state.approximation_degree = 0.99

failed = False
config = {}
with st.sidebar.form("configuration", clear_on_submit=False):
    st.header("Configuration", anchor=False)
    st.subheader("Backend", anchor=False)

    config["backend_choice"] = st.selectbox("Quantum system",
                                            options=range(engine.backend_db.size()),
                                            format_func=lambda id: backend_to_name(
                                                engine.backend_db[id]),
                                            index=0, help=descriptor["help_quantum_system"])

    config["shots"] = st.number_input("Shots", min_value=1, max_value=10 ** 5, step=1,
                                      value=st.session_state.shots, help=descriptor["help_shots"])
    st.divider()

    st.subheader("Input", anchor=False)
    secret_str = st.text_input("Secret string", value="1101", help=descriptor["help_secret_str"])
    secret_placeholder = st.empty()
    st.divider()

    st.subheader("Noise model", anchor=False)
    config["reset_err"] = st.slider("Reset error rate", min_value=0.0, max_value=0.1, format="%.3f",
                                    value=st.session_state.reset_rate, step=0.001,
                                    help=descriptor["help_reset_err"])
    config["meas_err"] = st.slider("Measure error rate", min_value=0.0, max_value=0.5,
                                   format="%.3f",
                                   value=st.session_state.measure_rate, step=0.001,
                                   help=descriptor["help_measurement_err"])
    config["single_err"] = st.slider("Single Gate error rate", min_value=0.0, max_value=0.5,
                                     format="%.3f",
                                     value=st.session_state.single_gate_rate,
                                     step=0.001,
                                     help=descriptor["help_single_gate_err"])
    config["double_err"] = st.slider("Two Gate error rate", min_value=0.0, max_value=0.5,
                                     format="%.3f",
                                     value=st.session_state.double_gate_rate, step=0.001,
                                     help=descriptor["help_double_gate_err"])
    st.divider()

    st.subheader("Transpiler", anchor=False)
    config["layout"] = st.selectbox("Layout method", options=[lm.value for lm in LayoutMethod],
                                    index=2,
                                    format_func=method_to_name,
                                    help=descriptor["help_layout_method"])
    config["routing"] = st.selectbox("Routing method", options=[rm.value for rm in RoutingMethod],
                                     index=1, format_func=method_to_name,
                                     help=descriptor["help_routing_method"])
    config["translation"] = st.selectbox("Translation method",
                                         options=[tm.value for tm in TranslationMethod],
                                         index=1, format_func=method_to_name,
                                         help=descriptor["help_translation_method"])
    config["optimization"] = st.select_slider("Optimization level", value=1,
                                              options=[ol.value for ol in OptimizationLevel],
                                              format_func=optimization_to_name,
                                              help=descriptor["help_optimization_level"])
    config["approx"] = st.slider("Approximation degree", min_value=0.9, max_value=1.0,
                                 format="%.2f",
                                 value=st.session_state.approximation_degree, step=0.01,
                                 help=descriptor["help_approximation_degree"], )
    submitted = st.form_submit_button("Execute", type="primary", disabled=False,
                                      use_container_width=True)

# ======================================

engine.configure(config)

if len(secret_str) > engine.configuration.backend.num_qubits - 1:
    secret_placeholder.error(descriptor["err_secret_str_length"](
        str_len=len(secret_str),
        qu_num=engine.configuration.backend.num_qubits))
    failed = True
elif not all(c in '01' for c in secret_str):
    secret_placeholder.error(descriptor["err_secret_str_value"])
    failed = True
if failed:
    st.warning(descriptor["warn_failure"])
    st.stop()


# with st.spinner('Wait for it...'):
# result = engine.run(secret_str=secret_str)
# @st.cache_data
def run(_engine, config, secret):
    return _engine.run(secret_str=secret)


result = run(engine, config, secret_str)

# ======================================

solution_cols = st.columns(2)
with solution_cols[0]:
    st.caption(":orange[Classical] approach")
    sol_cols = st.columns(2)
    sol_cols[0].metric(":orange[CL] solution", value=result.cl_solution, delta="OK",
                       help=descriptor["help_classical_solution"])
    sol_cols[1].metric(":orange[CL] byte instructions", value=result.solver.ops_count())

    oracle_cols = st.columns(2)
    oracle_cols[0].metric(":orange[CL] duration", value=f"{result.cl_time} s")
    oracle_cols[1].metric(":orange[CL] queries count", value=f"{result.cl_oracle.query_count} x")

with solution_cols[1]:
    st.caption(":violet[Quantum] approach")
    ok = result.qu_solution == secret_str
    sol_cols = st.columns(2)
    sol_cols[0].metric(":violet[QU] solution", value=result.qu_solution,
                       delta="OK" if ok else "BAD",
                       delta_color="normal" if ok else "inverse",
                       help=descriptor["help_quantum_solution"])
    sol_cols[1].metric(":violet[QU] shots", value=result.configuration.shot_count)

    oracle_cols = st.columns(2)
    oracle_cols[0].metric(":violet[QU] duration", value=f"{result.qu_time} s")
    oracle_cols[1].metric(":violet[QU] queries count", value=f"{result.qu_oracle.query_count} x")
st.divider()

st.header("Quantum hardware", anchor=False)
backend_cols = st.columns(2)
with backend_cols[0]:
    st.subheader("Backend metrics", anchor=False)

    cols1 = st.columns(2)
    cols1[0].metric("Classical bits", value=f"{result.builder.circuit.num_clbits} b",
                    help=descriptor["help_cl_bits"])
    cols1[1].metric("Quantum bits", value=f"{result.builder.circuit.size()} qu",
                    help=descriptor["help_qu_bits"])

    cols2 = st.columns(2)
    cols2[0].metric("Quantum gates", value=f"{result.builder.circuit.num_qubits}",
                    help=descriptor["help_qu_gates"])
    cols2[1].metric("Quantum bits (cap)", value=f"{result.configuration.backend.num_qubits} qu",
                    help=descriptor["help_qu_bits_cap"])

    if result.job.result().success:
        st.caption(f"{result.job.backend()} {result.sim.backend.backend_version} (:green[success])")
    else:
        st.caption(f"{result.job.backend()} {result.sim.backend.backend_version} (':red[fail]')")

with backend_cols[1]:
    qu_used_gates = result.builder.circuit.count_ops()
    gates = {"instruction": [], "count": []}
    for instruction, count in qu_used_gates.items():
        gates["instruction"].append(instruction)
        gates["count"].append(count)
    df = pd.DataFrame.from_dict(gates)
    st.dataframe(df, use_container_width=True)
st.divider()

gate_cols = st.columns([2, 3])
with gate_cols[0]:
    tabs = st.tabs(["Transpiled circuit layout", "Device's gate map"])
    fig1 = plot_circuit_layout(result.sim.compiled_circuit, result.sim.backend)
    fig2 = plot_gate_map(result.sim.backend, label_qubits=True)
    tabs[0].pyplot(fig1, clear_figure=True)
    tabs[1].pyplot(fig2, clear_figure=True)

with gate_cols[1]:
    st.subheader("Circuit layout", anchor=False)
    st.write(descriptor["text_circuit_layout"])
    st.caption(f"transpiler seed: :blue[{result.configuration.transpiler_seed}]")

gate_cols2 = st.columns([2, 3])
with gate_cols2[0]:
    st.subheader("Error map", anchor=False)
    st.write(descriptor["text_error_map"])
    fig = plot_error_map(result.sim.backend, figsize=(12, 10), show_title=False)
    gate_cols2[1].pyplot(fig, clear_figure=True)
st.divider()

st.header("Quantum circuit", anchor=False)
st.write(descriptor["text_quantum_circuit"])
circuit_tabs = st.tabs(["Built circuit", "Compiled circuit"])
fig = result.builder.circuit.draw(output="mpl", scale=1.1, justify="left", fold=-1,
                                  initial_state=False, plot_barriers=True,
                                  idle_wires=True, with_layout=True, cregbundle=True)
circuit_tabs[0].pyplot(fig, clear_figure=True)
fig = result.sim.compiled_circuit.draw(output="mpl", scale=1, justify="left", fold=-1,
                                       initial_state=False, plot_barriers=True,
                                       idle_wires=False, with_layout=False, cregbundle=True)
circuit_tabs[1].pyplot(fig, clear_figure=True)
st.divider()

st.header("Measurements", anchor=False)
st.write(descriptor["text_measurements"])

xs = np.asarray(list(result.counts.keys()))
ys = np.asarray(list(result.counts.values()))
xs, ys = sort_zipped(xs, ys)
sol_pos = 0
for pos, x in enumerate(xs):
    if secret_str == x:
        sol_pos = pos
        break

fig = plt.figure(figsize=(12, 6), dpi=200)
ax1 = fig.add_subplot(1, 1, 1)
bar = ax1.bar(xs, ys, color="#6b6b6b")
bar[sol_pos].set_color("#8210d8")
ax1.grid(axis='y', color='grey', linewidth=0.5, alpha=0.3)
ax1.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
ax1.set_xlabel("Binary measurement", fontsize=13, labelpad=20)
ax1.set_ylabel('Count', fontsize=13, labelpad=20)
other_patch = Patch(color='#6b6b6b', label='noise')
secret_patch = Patch(color='#8210d8', label='target')
fig.legend(handles=[other_patch, secret_patch])
st.pyplot(fig, clear_figure=True)
st.caption(f"simulator seed: :blue[{result.configuration.simulator_seed}]")
st.divider()

pie_cols = st.columns([1, 3])
with pie_cols[0]:
    st.subheader("Error rate", anchor=False)
    st.write(descriptor["text_error_rate"])

with pie_cols[1]:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5), dpi=200)
    fig.subplots_adjust(wspace=0)

    st.write()
    st.write()

    correct = result.counts[secret_str]
    incorrect = result.configuration.shot_count - result.counts[secret_str]
    total = result.configuration.shot_count

    overall_ratios = [incorrect / total, correct / total]
    labels = ['noise', 'target']
    explode = [0, 0.1]
    angle = -180 * overall_ratios[0]

    counts = {i: 0 for i in range(1, len(secret_str) + 1)}
    for meas in result.measurements:
        if meas != secret_str:
            counts[diff_letters(meas, secret_str)] += 1

    correct_ratios = np.asarray([counts[i] for i in range(1, len(secret_str) + 1)])
    correct_ratios = correct_ratios / sum(correct_ratios)
    correct_labels = [f"{i} qu" for i in range(1, len(secret_str) + 1)]
    bottom = 1
    width = .2


    def pct_func(pct, total):
        absolute = int(np.round(pct * total / 100))
        return f"{pct:.2f}%\n({absolute:d} shots)"


    wedges, *_ = ax1.pie(overall_ratios, autopct=lambda pct: pct_func(pct, total), startangle=angle, labels=labels,
                         explode=explode, colors=["#6b6b6b", "#8210d8"], textprops=dict(color="w"))
    ax1.legend(wedges, labels, title="Measurements", loc="upper left")

    # Adding from the top matches the legend.
    for j, (height, label) in enumerate(reversed([*zip(correct_ratios, correct_labels)])):
        bottom -= height
        bc = ax2.bar(0, height, width, bottom=bottom, color='#eb4034', label=label, alpha=0.1 + j / len(secret_str))
        ax2.bar_label(bc, labels=[f"{height:.0%}"], label_type='center')

    ax2.set_title('Number of incorrect qubits', pad=15, loc="right")
    ax2.legend()
    ax2.axis('off')
    ax2.set_xlim(- 2.5 * width, 2.5 * width)
    fig.tight_layout()

    theta1, theta2 = wedges[0].theta1, wedges[0].theta2
    center, r = wedges[0].center, wedges[0].r
    bar_height = sum(correct_ratios)

    # draw top connecting line
    x = r * np.cos(np.pi / 180 * theta2) + center[0]
    y = r * np.sin(np.pi / 180 * theta2) + center[1]
    con = ConnectionPatch(xyA=(-width / 2, bar_height), coordsA=ax2.transData,
                          xyB=(x, y), coordsB=ax1.transData)
    con.set_color("#000000")
    con.set_linewidth(0.6)
    ax2.add_artist(con)

    # draw bottom connecting line
    x = r * np.cos(np.pi / 180 * theta1) + center[0]
    y = r * np.sin(np.pi / 180 * theta1) + center[1]
    con = ConnectionPatch(xyA=(-width / 2, 0), coordsA=ax2.transData,
                          xyB=(x, y), coordsB=ax1.transData)
    con.set_color("#000000")
    ax2.add_artist(con)
    con.set_linewidth(0.6)

    st.pyplot(fig)
st.divider()

download_cols = st.columns(4)
timestamp = timestamp_str()
download_cols[0].subheader("Downloads:", anchor=False)
qu_qasm = QuantumCircuitBuild() \
    .create_circuit(oracle=result.qu_oracle, random_initialization=False) \
    .circuit.qasm(formatted=False)
download_cols[1].download_button("OpenQASM (qasm)", data=qu_qasm, mime="text/plain",
                                 help=descriptor["help_openqasm"], use_container_width=True,
                                 file_name=f"bernstein_vazirani_{timestamp}.qasm")
memory_csv = '\n'.join(result.measurements)
download_cols[2].download_button("Measurements (CSV)", data=memory_csv, mime="text/csv",
                                 help=descriptor["help_measurement_csv"],
                                 use_container_width=True,
                                 file_name=f"bernstein_vazirani_{timestamp}.csv")
counts_json = json.dumps(result.counts, indent=2, sort_keys=True)
download_cols[3].download_button("Counts (JSON)", data=counts_json, mime="application/json",
                                 help=descriptor["help_counts_json"], use_container_width=True,
                                 file_name=f"bernstein_vazirani_{timestamp}.json")

# yss = result.measurements
# ys = [int(i, 2) for i in yss]
# xs = list(range(len(ys)))
#
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.scatter(xs, ys, alpha=0.1)
# ax.tick_params(
#     axis='y',
#     which='both',
#     labelleft=False)
# st.pyplot(fig, clear_figure=True)

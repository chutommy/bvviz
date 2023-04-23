import json

import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from qiskit.visualization import plot_circuit_layout, plot_gate_map, plot_error_map

from bernstein_vazirani import QuantumCircuitBuild
from data import Descriptor
from engine import Engine
from page import render_sidebar, init_session_state, render_secret_check, render_basic_metrics, render_quantum_hardware
from utils import timestamp_str, sort_zipped, diff_letters

descriptor = Descriptor('assets/descriptions.json')
engine = Engine()
config = {}

init_session_state(descriptor)
st.title("Bernstein–Vazirani Quantum Protocol", anchor=False)
st.divider()
secret_str, secret_placeholder = render_sidebar(engine, config, descriptor)
engine.configure(config)
render_secret_check(engine, descriptor, secret_str, secret_placeholder)
result = engine.run(secret_str)

# ======================================

quantum_hardware_proc = {}
gates = {"instruction": [], "count": []}
for instruction, count in result.builder.circuit.count_ops().items():
    gates["instruction"].append(instruction)
    gates["count"].append(count)
quantum_hardware_proc["gates"] = gates
quantum_hardware_proc["fig1"] = plot_circuit_layout(result.sim.compiled_circuit, result.sim.backend)
quantum_hardware_proc["fig2"] = plot_gate_map(result.sim.backend, label_qubits=True)
quantum_hardware_proc["fig3"] = plot_error_map(result.sim.backend, figsize=(12, 10), show_title=False)
quantum_hardware_proc["fig4"] = result.builder.circuit.draw(output="mpl", scale=1.1, justify="left", fold=-1,
                                                            initial_state=False, plot_barriers=True,
                                                            idle_wires=True, with_layout=True, cregbundle=True)
quantum_hardware_proc["fig5"] = result.sim.compiled_circuit.draw(output="mpl", scale=1, justify="left", fold=-1,
                                                                 initial_state=False, plot_barriers=True,
                                                                 idle_wires=False, with_layout=False, cregbundle=True)

# ======================================

render_basic_metrics(result, descriptor)
st.divider()
render_quantum_hardware(result, descriptor, quantum_hardware_proc)
st.divider()

st.header("Measurements", anchor=False)

meas_tabs = st.tabs(["Counts", "Measurements"])
with meas_tabs[1]:
    ys = [int(i, 2) for i in result.measurements]
    xs = list(range(len(ys)))

    fig = plt.figure(figsize=(12, 6), dpi=200)
    ax = fig.add_subplot(1, 1, 1)
    scat = ax.scatter(ys, xs, alpha=0.1, color="#8210d8")
    ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax.set_xlabel("Binary measurement", fontsize=13, labelpad=20)
    ax.set_ylabel('Measurement number', fontsize=13, labelpad=20)
    st.pyplot(fig, clear_figure=True)

with meas_tabs[0]:
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
    ax1.grid(axis='y', color='grey', linewidth=0.5, alpha=0.4)
    ax1.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax1.set_xlabel("Binary measurement", fontsize=13, labelpad=20)
    ax1.set_ylabel('Count', fontsize=13, labelpad=20)
    other_patch = Patch(color='#6b6b6b', label='noise')
    secret_patch = Patch(color='#8210d8', label='target')
    ax1.legend(handles=[other_patch, secret_patch])
    st.pyplot(fig, clear_figure=True)

st.write(descriptor["text_measurements"])
st.divider()

correct = result.counts[secret_str]
incorrect = result.configuration.shot_count - result.counts[secret_str]
total = result.configuration.shot_count

counts = np.array(list(result.counts.values()))

meas_cols = st.columns(2)
with meas_cols[0]:
    st.subheader("Metrics")
    metric_cols = st.columns(2)

    with metric_cols[0]:
        st.metric(":blue[Correct] rate", value=f"{correct / total * 100:.2f} %")
        st.metric(":blue[Confidence] level", value="max likelihood" if incorrect == 0 else f"{correct / incorrect:.2f}")

    with metric_cols[1]:
        st.metric(":red[Error] rate (normalized)",
                  value=f"{2 * incorrect / total / (2 ** len(secret_str) - 1) * 100:.2f} %")
        st.metric(":red[Error] rate (total)", value=f"{incorrect / total * 100:.2f} %")

    st.caption(f"simulator seed: :blue[{result.configuration.simulator_seed}]")

with meas_cols[1]:
    fig = plt.figure(figsize=(12, 5), dpi=200)
    ax1 = fig.add_subplot(1, 1, 1)
    bar = ax1.bar(xs, ys, color="#6b6b6b")
    bar[sol_pos].set_color("#8210d8")
    ax1.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax1.axhline(y=(counts.mean() + counts.max(initial=0.5)) / 2, color='r', linestyle='-')
    st.pyplot(fig, clear_figure=True)
st.divider()

pie_cols = st.columns([2, 1])
with pie_cols[1]:
    st.subheader("Error rate", anchor=False)
    st.write(descriptor["text_error_rate"])

with pie_cols[0]:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5), dpi=200)
    fig.subplots_adjust()

    st.write()
    st.write()

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

    ax2.axis('off')
    # if incorrect != 0:
    #     for j, (height, label) in enumerate(reversed([*zip(correct_ratios, correct_labels)])):
    #         bottom -= height
    #         bc = ax2.bar(0, height, width, bottom=bottom, color='#eb4034', label=label,
    #                      alpha=0.1 + j / (len(secret_str) + 1))
    #         ax2.bar_label(bc, labels=[f"{height:.0%}"], label_type='center')
    #
    #     ax2.set_title('Number of incorrect qubits', pad=15, loc="right")
    #     ax2.legend()
    #     ax2.set_xlim(- 2.5 * width, 2.5 * width)
    #
    #     if correct != 0:
    #         theta1, theta2 = wedges[0].theta1, wedges[0].theta2
    #         center, r = wedges[0].center, wedges[0].r
    #         bar_height = sum(correct_ratios)
    #
    #         # draw top connecting line
    #         x = r * np.cos(np.pi / 180 * theta2) + center[0]
    #         y = r * np.sin(np.pi / 180 * theta2) + center[1]
    #         con = ConnectionPatch(xyA=(-width / 2, bar_height), coordsA=ax2.transData,
    #                               xyB=(x, y), coordsB=ax1.transData)
    #         con.set_color("#000000")
    #         con.set_linewidth(0.6)
    #         ax2.add_artist(con)
    #
    #         # draw bottom connecting line
    #         x = r * np.cos(np.pi / 180 * theta1) + center[0]
    #         y = r * np.sin(np.pi / 180 * theta1) + center[1]
    #         con = ConnectionPatch(xyA=(-width / 2, 0), coordsA=ax2.transData,
    #                               xyB=(x, y), coordsB=ax1.transData)
    #         con.set_color("#000000")
    #         ax2.add_artist(con)
    #         con.set_linewidth(0.6)

    st.pyplot(fig)

st.subheader("Downloads:", anchor=False)
timestamp = timestamp_str()

qu_qasm = QuantumCircuitBuild() \
    .create_circuit(oracle=result.qu_oracle, random_initialization=False) \
    .circuit.qasm(formatted=False)
st.download_button("OpenQASM (qasm)", data=qu_qasm, mime="text/plain",
                   help=descriptor["help_openqasm"], use_container_width=True,
                   file_name=f"bernstein_vazirani_{timestamp}.qasm")

counts_json = json.dumps(result.counts, indent=2, sort_keys=True)
st.download_button("Counts (JSON)", data=counts_json, mime="application/json",
                   help=descriptor["help_counts_json"], use_container_width=True,
                   file_name=f"bernstein_vazirani_{timestamp}.json")

memory_csv = '\n'.join(result.measurements)
st.download_button("Measurements (CSV)", data=memory_csv, mime="text/csv",
                   help=descriptor["help_measurement_csv"],
                   use_container_width=True,
                   file_name=f"bernstein_vazirani_{timestamp}.csv")

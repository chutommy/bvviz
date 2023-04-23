"""Handles web page rendering."""

import streamlit as st
import streamlit_ext as ste
from streamlit.delta_generator import DeltaGenerator

from .config import LayoutMethod, RoutingMethod, TranslationMethod, OptimizationLevel
from .data import Descriptor
from .engine import Engine, Result
from .utils import backend_to_name, method_to_name, optimization_to_name, generate_seed, \
    check_secret


def init_session_state():
    """Initialize web page session."""
    if 'init' not in st.session_state:
        st.session_state.init = True

        st.session_state.backend_choice = 0
        st.session_state.layout_method = 2
        st.session_state.routing_method = 1
        st.session_state.translation_method = 1
        st.session_state.secret = "11010"

        st.session_state.transpiler_seed = generate_seed()
        st.session_state.simulator_seed = generate_seed()
        st.session_state.shots = 1000

        st.session_state.reset_rate = 0.019
        st.session_state.measure_rate = 0.154
        st.session_state.single_gate_rate = 0.146
        st.session_state.double_gate_rate = 0.179

        st.session_state.approximation_degree = 0.99
        st.session_state.optimization_level = 1


def render_sidebar(eng: Engine, cfg: dict, des: Descriptor) -> (str, DeltaGenerator):
    """Renders sidebar."""
    with st.sidebar.form("configuration", clear_on_submit=False):
        st.header("Configuration", anchor=False)
        st.subheader("Backend", anchor=False)

        cfg["backend_choice"] = st.selectbox("Quantum system", options=range(eng.backend_db.size()),
                                             format_func=lambda key: backend_to_name(
                                                 eng.backend_db[key]),
                                             index=st.session_state.backend_choice,
                                             help=des["help_quantum_system"])

        cfg["shots"] = st.number_input("Shots", min_value=1, max_value=10 ** 5, step=1,
                                       value=st.session_state.shots, help=des["help_shots"])
        st.divider()

        st.subheader("Input", anchor=False)
        secret_str = st.text_input("Secret string", value=st.session_state.secret,
                                   help=des["help_secret_str"])
        secret_placeholder = st.empty()
        st.divider()

        st.subheader("Noise model", anchor=False)
        cfg["reset_err"] = st.slider("Reset error rate", min_value=0.0, max_value=0.1,
                                     format="%.3f",
                                     value=st.session_state.reset_rate, step=0.001,
                                     help=des["help_reset_err"])
        cfg["meas_err"] = st.slider("Measure error rate", min_value=0.0, max_value=0.5,
                                    format="%.3f",
                                    value=st.session_state.measure_rate, step=0.001,
                                    help=des["help_measurement_err"])
        cfg["single_err"] = st.slider("Single Gate error rate", min_value=0.0, max_value=0.5,
                                      format="%.3f",
                                      value=st.session_state.single_gate_rate,
                                      step=0.001,
                                      help=des["help_single_gate_err"])
        cfg["double_err"] = st.slider("Two Gate error rate", min_value=0.0, max_value=0.5,
                                      format="%.3f",
                                      value=st.session_state.double_gate_rate, step=0.001,
                                      help=des["help_double_gate_err"])
        st.divider()

        st.subheader("Transpiler", anchor=False)
        cfg["layout"] = st.selectbox("Layout method", options=[lm.value for lm in LayoutMethod],
                                     index=st.session_state.layout_method,
                                     format_func=method_to_name,
                                     help=des["help_layout_method"])
        cfg["routing"] = st.selectbox("Routing method", options=[rm.value for rm in RoutingMethod],
                                      index=st.session_state.routing_method,
                                      format_func=method_to_name,
                                      help=des["help_routing_method"])
        cfg["translation"] = st.selectbox("Translation method",
                                          options=[tm.value for tm in TranslationMethod],
                                          index=st.session_state.translation_method,
                                          format_func=method_to_name,
                                          help=des["help_translation_method"])
        cfg["optimization"] = st.select_slider("Optimization level",
                                               value=st.session_state.optimization_level,
                                               options=[ol.value for ol in OptimizationLevel],
                                               format_func=optimization_to_name,
                                               help=des["help_optimization_level"])
        cfg["approx"] = st.slider("Approximation degree", min_value=0.9, max_value=1.0,
                                  format="%.2f",
                                  value=st.session_state.approximation_degree, step=0.01,
                                  help=des["help_approximation_degree"], )
        cfg["submitted"] = st.form_submit_button("Execute", type="primary", disabled=False,
                                                 use_container_width=True)

        return secret_str, secret_placeholder


def render_secret_check(eng: Engine, des: Descriptor, secret: str, placeholder: DeltaGenerator):
    """Renders secret warning on invalid secret."""
    if eng.check_secret_size(secret):
        placeholder.error(des["err_secret_str_length"](str_len=len(secret),
                                                       qu_num=eng.configuration.backend.num_qubits))
        st.warning(des["warn_failure"])
        st.stop()
    elif check_secret(secret):
        placeholder.error(des["err_secret_str_value"])
        st.warning(des["warn_failure"])
        st.stop()


def render_basic_metrics(res: Result, des: Descriptor):
    """Renders basic metrics section."""
    cols = st.columns(2)

    with cols[0]:
        st.caption(":orange[Classical] approach")

        cols1 = st.columns(2)
        cols1[0].metric(":orange[CL] solution", value=res.cl_result.solution, delta="OK",
                        help=des["help_classical_solution"])
        cols1[1].metric(":orange[CL] byte instructions", value=res.snap.solver.ops_count())

        cols2 = st.columns(2)
        cols2[0].metric(":orange[CL] time", value=f"{res.cl_result.time} s")
        cols2[1].metric(":orange[CL] queries count", value=f"{res.cl_result.oracle.query_count} x")

    with cols[1]:
        st.caption(":violet[Quantum] approach")
        good_solution = res.qu_result.solution == res.secret
        cols1 = st.columns(2)
        cols1[0].metric(":violet[QU] solution", value=res.qu_result.solution,
                        delta="OK" if good_solution else "BAD",
                        delta_color="normal" if good_solution else "inverse",
                        help=des["help_quantum_solution"])
        cols1[1].metric(":violet[QU] shots", value=res.snap.configuration.shot_count)

        cols2 = st.columns(2)
        cols2[0].metric(":violet[QU] time", value=f"{res.qu_result.time} s")
        cols2[1].metric(":violet[QU] queries count", value=f"{res.qu_result.oracle.query_count} x")


def render_quantum_hardware(res: Result, des: Descriptor, quantum_hardware_proc: dict):
    """Renders quantum hardware section."""
    st.header("Quantum hardware", anchor=False)
    backend_cols = st.columns(2)

    with backend_cols[0]:
        st.subheader("Backend metrics", anchor=False)

        cols1 = st.columns(2)
        cols1[0].metric("Classical bits", value=f"{res.snap.builder.circuit.num_clbits} b",
                        help=des["help_cl_bits"])
        cols1[1].metric("Quantum bits", value=f"{res.snap.builder.circuit.size()} qu",
                        help=des["help_qu_bits"])

        cols2 = st.columns(2)
        cols2[0].metric("Quantum gates", value=f"{res.snap.builder.circuit.num_qubits}",
                        help=des["help_qu_gates"])
        cols2[1].metric("Quantum bits (cap)",
                        value=f"{res.snap.configuration.backend.num_qubits} qu",
                        help=des["help_qu_bits_cap"])

        status_message = ':green[success]' if res.result.success else ':red[fail]'
        # noinspection PyUnresolvedReferences
        st.caption(f"{res.job.backend()} {res.snap.sim.backend.backend_version} ({status_message})")

    backend_cols[1].table(quantum_hardware_proc["gates"])
    st.divider()

    gate_cols = st.columns([2, 3])
    with gate_cols[0]:
        tabs = st.tabs(["Transpiled circuit layout", "Device's gate map"])
        tabs[0].pyplot(quantum_hardware_proc["layout_circuit"], clear_figure=True)
        tabs[1].pyplot(quantum_hardware_proc["map_gate"], clear_figure=True)

    with gate_cols[1]:
        st.subheader("Circuit layout", anchor=False)
        st.write(des["text_circuit_layout"])
        st.caption(f"transpiler seed: :blue[{res.snap.configuration.transpiler_seed}]")

    gate_cols2 = st.columns([2, 3])
    with gate_cols2[0]:
        st.subheader("Error map", anchor=False)
        st.write(des["text_error_map"])
        gate_cols2[1].pyplot(quantum_hardware_proc["map_error"], clear_figure=True)
    st.divider()

    st.header("Quantum circuit", anchor=False)
    st.write(des["text_quantum_circuit"])
    circuit_tabs = st.tabs(["Built circuit", "Compiled circuit"])
    circuit_tabs[0].pyplot(quantum_hardware_proc["circuit"], clear_figure=True)
    circuit_tabs[1].pyplot(quantum_hardware_proc["circuit_compiled"], clear_figure=True)


def render_measurement(res: Result, des: Descriptor, proc: dict):
    """Renders measurement section."""
    st.header("Measurements", anchor=False)

    meas_tabs = st.tabs(["Counts", "Measurements"])
    meas_tabs[0].pyplot(proc['bar_counts'], clear_figure=True)
    meas_tabs[1].pyplot(proc['scatter_counts'], clear_figure=True)

    st.write(des["text_measurements"])
    st.divider()

    meas_cols = st.columns(2)
    with meas_cols[0]:
        st.subheader("Metrics")
        metric_cols = st.columns(2)
        metric_cols[0].metric(":blue[Correct] rate", value=proc['correct_rate'])
        metric_cols[0].metric(":blue[Confidence] level", value=proc['confidence_level'])
        metric_cols[1].metric(":red[Error] rate (normalized)",
                              value=proc['error_rate_norm'])
        metric_cols[1].metric(":red[Error] rate (total)",
                              value=proc['error_rate_total'])
        st.caption(f"simulator seed: :blue[{res.snap.configuration.simulator_seed}]")
    meas_cols[1].pyplot(proc['bar_counts_minimal'], clear_figure=True)
    st.divider()

    pie_cols = st.columns([2, 1])
    pie_cols[1].subheader("Error rate", anchor=False)
    pie_cols[1].write(des["text_error_rate"])
    pie_cols[0].pyplot(proc['pie_error_rate'])


def render_download_buttons(des: Descriptor, proc: dict):
    """Renders measurement section."""
    st.subheader("Downloads:", anchor=False)
    ste.download_button("OpenQASM (qasm)", data=proc['qu_qasm'], mime="text/plain",
                        file_name=f"bernstein_vazirani_{proc['timestamp']}.qasm")
    ste.download_button("Counts (JSON)", data=proc['counts_json'], mime="application/json",
                        file_name=f"bernstein_vazirani_{proc['timestamp']}.json")
    ste.download_button("Measurements (CSV)", data=proc['memory_csv'], mime="text/csv",
                        file_name=f"bernstein_vazirani_{proc['timestamp']}.csv")

    _ = des
    # st.download_button("OpenQASM (qasm)", data=measurement_proc['qu_qasm'], mime="text/plain",
    #                    help=des["help_openqasm"], use_container_width=True,
    #                    file_name=f"bernstein_vazirani_{measurement_proc['timestamp']}.qasm")
    # st.download_button("Counts (JSON)", data=measurement_proc['counts_json'],
    #                    mime="application/json",
    #                    help=des["help_counts_json"], use_container_width=True,
    #                    file_name=f"bernstein_vazirani_{measurement_proc['timestamp']}.json")
    # st.download_button("Measurements (CSV)", data=measurement_proc['memory_csv'],
    #                    mime="text/csv",
    #                    help=des["help_measurement_csv"],
    #                    use_container_width=True,
    #                    file_name=f"bernstein_vazirani_{measurement_proc['timestamp']}.csv")

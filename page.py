import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from config import LayoutMethod, RoutingMethod, TranslationMethod, OptimizationLevel
from data import Descriptor
from engine import Engine, Result
from utils import backend_to_name, method_to_name, optimization_to_name, generate_seed, check_secret


def init_session_state(d: Descriptor):
    st.set_page_config(page_title="Bernstein–Vazirani", page_icon="assets/logo.png",
                       layout="wide", initial_sidebar_state="auto", menu_items=None)
    st.markdown(d.cat(["style_hide_header", "style_hide_footer", "style_hide_view_fullscreen"]), unsafe_allow_html=True)
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
        st.session_state.optimizatio_level = 1


def render_sidebar(e: Engine, cfg: dict, d: Descriptor) -> (str, DeltaGenerator):
    with st.sidebar.form("configuration", clear_on_submit=False):
        st.header("Configuration", anchor=False)
        st.subheader("Backend", anchor=False)

        cfg["backend_choice"] = st.selectbox("Quantum system", options=range(e.backend_db.size()),
                                             format_func=lambda id: backend_to_name(e.backend_db[id]),
                                             index=st.session_state.backend_choice,
                                             help=d["help_quantum_system"])

        cfg["shots"] = st.number_input("Shots", min_value=1, max_value=10 ** 5, step=1,
                                       value=st.session_state.shots, help=d["help_shots"])
        st.divider()

        st.subheader("Input", anchor=False)
        secret_str = st.text_input("Secret string", value=st.session_state.secret, help=d["help_secret_str"])
        secret_placeholder = st.empty()
        st.divider()

        st.subheader("Noise model", anchor=False)
        cfg["reset_err"] = st.slider("Reset error rate", min_value=0.0, max_value=0.1, format="%.3f",
                                     value=st.session_state.reset_rate, step=0.001,
                                     help=d["help_reset_err"])
        cfg["meas_err"] = st.slider("Measure error rate", min_value=0.0, max_value=0.5,
                                    format="%.3f",
                                    value=st.session_state.measure_rate, step=0.001,
                                    help=d["help_measurement_err"])
        cfg["single_err"] = st.slider("Single Gate error rate", min_value=0.0, max_value=0.5,
                                      format="%.3f",
                                      value=st.session_state.single_gate_rate,
                                      step=0.001,
                                      help=d["help_single_gate_err"])
        cfg["double_err"] = st.slider("Two Gate error rate", min_value=0.0, max_value=0.5,
                                      format="%.3f",
                                      value=st.session_state.double_gate_rate, step=0.001,
                                      help=d["help_double_gate_err"])
        st.divider()

        st.subheader("Transpiler", anchor=False)
        cfg["layout"] = st.selectbox("Layout method", options=[lm.value for lm in LayoutMethod],
                                     index=st.session_state.layout_method,
                                     format_func=method_to_name,
                                     help=d["help_layout_method"])
        cfg["routing"] = st.selectbox("Routing method", options=[rm.value for rm in RoutingMethod],
                                      index=st.session_state.routing_method, format_func=method_to_name,
                                      help=d["help_routing_method"])
        cfg["translation"] = st.selectbox("Translation method",
                                          options=[tm.value for tm in TranslationMethod],
                                          index=st.session_state.translation_method, format_func=method_to_name,
                                          help=d["help_translation_method"])
        cfg["optimization"] = st.select_slider("Optimization level", value=st.session_state.optimizatio_level,
                                               options=[ol.value for ol in OptimizationLevel],
                                               format_func=optimization_to_name,
                                               help=d["help_optimization_level"])
        cfg["approx"] = st.slider("Approximation degree", min_value=0.9, max_value=1.0,
                                  format="%.2f",
                                  value=st.session_state.approximation_degree, step=0.01,
                                  help=d["help_approximation_degree"], )
        st.form_submit_button("Execute", type="primary", disabled=False,
                              use_container_width=True)

        return secret_str, secret_placeholder


def render_secret_check(e: Engine, d: Descriptor, secret: str, placeholder: DeltaGenerator):
    if e.check_secret_size(secret):
        placeholder.error(d["err_secret_str_length"](str_len=len(secret), qu_num=e.configuration.backend.num_qubits))
        st.warning(d["warn_failure"])
        st.stop()
    elif check_secret(secret):
        placeholder.error(d["err_secret_str_value"])
        st.warning(d["warn_failure"])
        st.stop()


def render_basic_metrics(r: Result, d: Descriptor):
    cols = st.columns(2)

    with cols[0]:
        st.caption(":orange[Classical] approach")

        cols1 = st.columns(2)
        cols1[0].metric(":orange[CL] solution", value=r.cl_solution, delta="OK",
                        help=d["help_classical_solution"])
        cols1[1].metric(":orange[CL] byte instructions", value=r.solver.ops_count())

        cols2 = st.columns(2)
        cols2[0].metric(":orange[CL] time", value=f"{r.cl_time} s")
        cols2[1].metric(":orange[CL] queries count", value=f"{r.cl_oracle.query_count} x")

    with cols[1]:
        st.caption(":violet[Quantum] approach")
        ok = r.qu_solution == r.secret
        cols1 = st.columns(2)
        cols1[0].metric(":violet[QU] solution", value=r.qu_solution,
                        delta="OK" if ok else "BAD",
                        delta_color="normal" if ok else "inverse",
                        help=d["help_quantum_solution"])
        cols1[1].metric(":violet[QU] shots", value=r.configuration.shot_count)

        cols2 = st.columns(2)
        cols2[0].metric(":violet[QU] time", value=f"{r.qu_time} s")
        cols2[1].metric(":violet[QU] queries count", value=f"{r.qu_oracle.query_count} x")


def render_quantum_hardware(r: Result, d: Descriptor, proc: dict):
    st.header("Quantum hardware", anchor=False)
    backend_cols = st.columns(2)

    with backend_cols[0]:
        st.subheader("Backend metrics", anchor=False)

        cols1 = st.columns(2)
        cols1[0].metric("Classical bits", value=f"{r.builder.circuit.num_clbits} b",
                        help=d["help_cl_bits"])
        cols1[1].metric("Quantum bits", value=f"{r.builder.circuit.size()} qu",
                        help=d["help_qu_bits"])

        cols2 = st.columns(2)
        cols2[0].metric("Quantum gates", value=f"{r.builder.circuit.num_qubits}",
                        help=d["help_qu_gates"])
        cols2[1].metric("Quantum bits (cap)", value=f"{r.configuration.backend.num_qubits} qu",
                        help=d["help_qu_bits_cap"])

        status_message = ':green[success]' if r.result.success else ':red[fail]'
        st.caption(f"{r.job.backend()} {r.sim.backend.backend_version} ({status_message})")

    backend_cols[1].table(proc["gates"])
    st.divider()

    gate_cols = st.columns([2, 3])
    with gate_cols[0]:
        tabs = st.tabs(["Transpiled circuit layout", "Device's gate map"])
        tabs[0].pyplot(proc["fig1"], clear_figure=True)
        tabs[1].pyplot(proc["fig2"], clear_figure=True)

    with gate_cols[1]:
        st.subheader("Circuit layout", anchor=False)
        st.write(d["text_circuit_layout"])
        st.caption(f"transpiler seed: :blue[{r.configuration.transpiler_seed}]")

    gate_cols2 = st.columns([2, 3])
    with gate_cols2[0]:
        st.subheader("Error map", anchor=False)
        st.write(d["text_error_map"])
        gate_cols2[1].pyplot(proc["fig3"], clear_figure=True)
    st.divider()

    st.header("Quantum circuit", anchor=False)
    st.write(d["text_quantum_circuit"])
    circuit_tabs = st.tabs(["Built circuit", "Compiled circuit"])
    circuit_tabs[0].pyplot(proc["fig4"], clear_figure=True)
    circuit_tabs[1].pyplot(proc["fig5"], clear_figure=True)

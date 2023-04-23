"""Starting point of the web app."""

import streamlit as st

from .data import Descriptor
from .engine import Engine, preprocess
from .page import init_session_state, render_sidebar, render_secret_check, render_basic_metrics, \
    render_quantum_hardware, render_measurement, render_download_buttons


def run():
    """Runs the web app."""
    descriptor = Descriptor('assets/descriptions.json')
    engine = Engine()
    config = {}

    st.set_page_config(page_title="Bernstein–Vazirani", page_icon="assets/logo.png",
                       layout="wide", initial_sidebar_state="auto", menu_items=None)
    st.write(descriptor.cat(["style_hide_header",
                             "style_hide_footer",
                             "style_hide_view_fullscreen"]), unsafe_allow_html=True)

    init_session_state()
    st.title("Bernstein–Vazirani Quantum Protocol", anchor=False)
    st.divider()

    secret_str, secret_placeholder = render_sidebar(engine, config, descriptor)
    engine.configure(config)
    render_secret_check(engine, descriptor, secret_str, secret_placeholder)
    Res = engine.run(secret_str)
    proc = preprocess(Res)
    render_basic_metrics(Res, descriptor)
    st.divider()
    render_quantum_hardware(Res, descriptor, proc)
    st.divider()
    render_measurement(Res, descriptor, proc)
    st.divider()
    render_download_buttons(descriptor, proc)

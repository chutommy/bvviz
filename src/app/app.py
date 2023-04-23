"""Starting point of the web app."""

import streamlit as st

from .data import Descriptor
from .engine import Engine, preprocess
from .page import init_session_state, render_sidebar, render_secret_check, render_basic_metrics, \
    render_quantum_hardware, render_measurement, render_download_buttons
from .utils import dhash


@st.cache_data
def get_result(_engine, _descriptor, _config, secret_str, _secret_placeholder, config_hash):
    """Calculates a Result - possibly cache."""
    _ = config_hash
    _engine.configure(_config)
    render_secret_check(_engine, _descriptor, secret_str, _secret_placeholder)
    res = _engine.run(secret_str)
    return res


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
    result = get_result(engine, descriptor, config, secret_str, secret_placeholder, dhash(config))
    proc = preprocess(result)
    render_basic_metrics(result, descriptor)
    st.divider()
    render_quantum_hardware(result, descriptor, proc)
    st.divider()
    render_measurement(result, descriptor, proc)
    st.divider()
    render_download_buttons(descriptor, proc)

"""Starts the web app."""

import streamlit as st

from data import Descriptor
from engine import Engine, preprocess
from page import render_sidebar, init_session_state, render_secret_check, render_basic_metrics, \
    render_quantum_hardware, render_measurement

descriptor = Descriptor('assets/descriptions.json')
engine = Engine()
config = {}

init_session_state(descriptor)
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

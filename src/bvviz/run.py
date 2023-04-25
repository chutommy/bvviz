"""Starting point of the web app."""
from typing import Any, Dict

import streamlit as st
import streamlit_ext as ste

from .data import Descriptor
from .engine import Engine, preprocess
from .page import (init_session_state, render_basic_metrics, render_footer,
                   render_introduction, render_measurement, render_quantum_hardware,
                   render_secret_check, render_sidebar)


def run(descriptions_path: str) -> None:
    """Runs the web app."""
    descriptor = Descriptor(descriptions_path)
    engine = Engine()
    config: Dict[str, Any] = {}

    st.set_page_config(page_title='Bernstein–Vazirani', page_icon='assets/logo.png',
                       layout='wide', initial_sidebar_state='auto', menu_items=None)
    st.write(descriptor.cat(['style_hide_header',
                             'style_hide_footer',
                             'style_hide_view_fullscreen']), unsafe_allow_html=True)
    ste.set_width('84em')

    init_session_state()
    st.title('Bernstein–Vazirani Quantum Protocol', anchor=False)
    render_introduction(descriptor)
    st.divider()

    secret_str, secret_placeholder = render_sidebar(engine, config, descriptor)
    engine.configure(config)
    render_secret_check(engine, descriptor, secret_str, secret_placeholder)

    result = engine.run(secret_str)
    ctx = preprocess(result)
    render_basic_metrics(result, descriptor)
    st.divider()
    render_quantum_hardware(result, descriptor, ctx)
    st.divider()
    render_measurement(result, descriptor, ctx)
    st.divider()
    render_footer(descriptor, ctx)

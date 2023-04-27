"""This module implements the app initialization process."""

from json import loads
from typing import Any, Dict

import streamlit as st
import streamlit_ext as ste
from matplotlib import pyplot as plt

from .data import Descriptor
from .engine import Engine, preprocess
from .page import (init_session_state, render_basic_metrics, render_footer,
                   render_introduction, render_measurement, render_quantum_hardware,
                   render_secret_check, render_sidebar)


def run(settings_path: str) -> None:
    """Runs the web app."""
    with open(settings_path, 'r', encoding='utf-8') as file:
        settings_json = file.read()
    settings = loads(settings_json)
    descriptor = Descriptor(settings['descriptions'])
    engine = Engine()
    config: Dict[str, Any] = {}

    st.set_page_config(page_title=settings['title'], page_icon=settings['page_icon'],
                       layout='wide', initial_sidebar_state='auto', menu_items=None)
    st.write(descriptor.cat(['style_hide_header',
                             'style_hide_footer',
                             'style_hide_view_fullscreen']), unsafe_allow_html=True)
    ste.set_width(settings['page_width'])
    init_session_state(settings['init'])

    secret_str, secret_placeholder = render_sidebar(engine, config, descriptor)
    engine.configure(config)
    render_secret_check(engine, descriptor, secret_str, secret_placeholder)

    with st.spinner(descriptor['info_wait_compute']()):
        tmp = st.divider()
        result = engine.run(secret_str)
        ctx = preprocess(result)
        tmp.empty()

    with st.spinner(descriptor['info_wait_render']()):
        tmp = st.divider()
        st.title('Visualisation of :violet[Bernstein-Vazirani] Quantum Protocol', anchor=False)
        st.caption('by Tommy Chu')
        render_introduction(descriptor)
        st.divider()
        render_basic_metrics(result, descriptor)
        st.divider()
        render_quantum_hardware(result, descriptor, ctx)
        st.divider()
        render_measurement(result, descriptor, ctx)
        st.divider()
        render_footer(descriptor, ctx)
        plt.close('all')
        tmp.empty()

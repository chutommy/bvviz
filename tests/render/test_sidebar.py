import streamlit_mock


def test_sidebar():
    sm = streamlit_mock.StreamlitMock()

    results = sm.run("app.py")

    results = sm.get_results()
import pytest
from seleniumbase import BaseCase

configList = [
    {
        'system': 'Ge',
        'shots': 5413,
        'secret': '00110',
        'reset_err': 32,
        'measure_err': 91,
        'single_err': 64,
        'double_err': 89,
        'layout': 'Den',
        'routing': 'Ba',
        'translation': 'Unro',
        'optimization': 1000,
        'approximation': -32,
    },
    {
        'system': 'Mon',
        'shots': 2031,
        'secret': '00110',
        'reset_err': 131,
        'measure_err': 65,
        'single_err': 31,
        'double_err': 60,
        'layout': 'Den',
        'routing': 'Look',
        'translation': 'Unrol',
        'optimization': 1000,
        'approximation': -56,
    },
    {
        'system': 'Hano',
        'shots': 3214,
        'secret': '00110',
        'reset_err': 35,
        'measure_err': 12,
        'single_err': 70,
        'double_err': 23,
        'layout': 'Sab',
        'routing': 'Sto',
        'translation': 'Synth',
        'optimization': 1000,
        'approximation': -13,
    },
    {
        'system': 'Sydne',
        'shots': 3213,
        'secret': '00110',
        'reset_err': 15,
        'measure_err': 77,
        'single_err': 11,
        'double_err': 89,
        'layout': 'Noise',
        'routing': 'Ba',
        'translation': 'Tra',
        'optimization': 1000,
        'approximation': -43,
    },
]


class uiTest(BaseCase):
    def test_ui(self):
        self.open('http://localhost:8501/')
        self.maximize_window()

        self.assert_title('Bernstein-Vazirani')
        self.assert_no_404_errors()
        self.assert_no_js_errors()

        self.type('input[aria-label="Selected Almaden (20). System"]', 'Si')
        self.click('li[role="option"]')
        self.type('input[aria-label="Shots"', 1234)
        self.type('input[aria-label="Secret string"]', '1011100')

        self.scroll_into_view('div[class="stSlider"]')
        self.drag_and_drop_with_offset('div[aria-label="Reset error rate"]', 73, 0)
        self.drag_and_drop_with_offset('div[aria-label="Measure error rate"]', 46, 0)
        self.drag_and_drop_with_offset('div[aria-label="Single Gate error rate"]', 45, 0)
        self.drag_and_drop_with_offset('div[aria-label="Two Gate error rate"', 91, 0)

        self.scroll_into_view('div[aria-label="Two Gate error rate"]')
        self.type('input[aria-label="Selected Noise Adaptive. Layout method"]', 'Sa')
        self.click('li[role="option"]')
        self.type('input[aria-label="Selected Lookahead. Routing method"]', 'St')
        self.click('li[role="option"]')
        self.type('input[aria-label="Selected Translator. Translation method"]', 'Synthesis')
        self.click('li[role="option"]')

        self.scroll_into_view('button[kind="primaryFormSubmit"]')
        self.drag_and_drop_with_offset('div[aria-label="Optimization level"]', 100, 0)
        self.drag_and_drop_with_offset('div[aria-label="Approximation degree"]', -30, 0)
        self.wait_for_element_not_present('.stSpinner')
        self.click('button[kind="primaryFormSubmit"]')
        self.wait_for_element_not_present('.stSpinner')

        self.scroll_into_view('button[tabindex="0"]')
        self.click('button[tabindex="-1"]')
        self.click('button[tabindex="-1"]')

        self.scroll_into_view('a[download="bernstein_vazirani.qasm"]')
        self.click('a[download="bernstein_vazirani.qasm"]')
        self.assert_downloaded_file('bernstein_vazirani.qasm')
        self.delete_downloaded_file('bernstein_vazirani.qasm')
        self.click('a[download="bernstein_vazirani.json"]')
        self.assert_downloaded_file('bernstein_vazirani.json')
        self.delete_downloaded_file('bernstein_vazirani.json')
        self.click('a[download="bernstein_vazirani.csv"]')
        self.assert_downloaded_file('bernstein_vazirani.csv')
        self.delete_downloaded_file('bernstein_vazirani.csv')

        self.click_link('Qiskit')
        self.assert_url_contains('qiskit.org')
        self.switch_to_default_window()

        self.click_link('Streamlit')
        self.assert_url_contains('streamlit.io')
        self.switch_to_default_window()

        self.click_link('Tommy Chu')
        self.assert_url_contains('github.com/chutommy')
        self.switch_to_default_window()

    def test_experiments(self):
        for config in configList:
            self.open('http://localhost:8501/')
            self.maximize_window()

            self.type('input[aria-label="Selected Almaden (20). System"]', config['system'])
            self.click('li[role="option"]')
            self.type('input[aria-label="Shots"', config['shots'])
            self.type('input[aria-label="Secret string"]', config['secret'])

            self.scroll_into_view('div[class="stSlider"]')
            self.drag_and_drop_with_offset('div[aria-label="Reset error rate"]',
                                           config['reset_err'], 0)
            self.drag_and_drop_with_offset('div[aria-label="Measure error rate"]',
                                           config['measure_err'], 0)
            self.drag_and_drop_with_offset('div[aria-label="Single Gate error rate"]',
                                           config['single_err'], 0)
            self.drag_and_drop_with_offset('div[aria-label="Two Gate error rate"',
                                           config['double_err'], 0)

            self.scroll_into_view('div[aria-label="Two Gate error rate"]')
            self.type('input[aria-label="Selected Noise Adaptive. Layout method"]',
                      config['layout'])
            self.click('li[role="option"]')
            self.type('input[aria-label="Selected Lookahead. Routing method"]', config['routing'])
            self.click('li[role="option"]')
            self.type('input[aria-label="Selected Translator. Translation method"]',
                      config['translation'])
            self.click('li[role="option"]')

            self.scroll_into_view('button[kind="primaryFormSubmit"]')
            self.drag_and_drop_with_offset('div[aria-label="Optimization level"]',
                                           config['optimization'], 0)
            self.drag_and_drop_with_offset('div[aria-label="Approximation degree"]',
                                           -config['approximation'], 0)
            self.wait_for_element_not_present('.stSpinner')
            self.click('button[kind="primaryFormSubmit"]')
            self.wait_for_element_not_present('.stSpinner')

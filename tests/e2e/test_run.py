from seleniumbase import BaseCase


class e2eTest(BaseCase):
    def test_run(self):
        self.open('http://localhost:8080/')
        # self.activate_demo_mode()
        self.maximize_window()

        self.assert_title('Bernstein–Vazirani')
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

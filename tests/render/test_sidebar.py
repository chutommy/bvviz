from seleniumbase import BaseCase


class UITester(BaseCase):
    # todo: https://github.com/streamlit/streamlit/issues/6482
    def test_swag_labs(self):
        self.open("http://localhost:8080/")

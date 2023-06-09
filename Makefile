lint-src:
	pylint --load-plugins pylint_quotes src/bvviz/
	mypy --ignore-missing-imports src/bvviz/

lint-test:
	pylint --load-plugins pylint_quotes --disable=C0114,C0116 tests/
	mypy --ignore-missing-imports tests/

lint: lint-src lint-test

test-unit:
	pytest --cov=src/bvviz -v -s --ignore=tests/ui tests/

test-ui:
	pytest -k uiTest --headed

test: test-unit test-ui

run:
	streamlit run app.py
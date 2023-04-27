lint:
	pylint --load-plugins pylint_quotes src/bvviz/
	mypy --strict --ignore-missing-imports --disable-error-code no-untyped-call src/bvviz/

	pylint --load-plugins pylint_quotes --disable=C0114,C0116 tests/
	mypy --strict --ignore-missing-imports --disable-error-code no-untyped-def --disable-error-code no-untyped-call tests/

test:
	pytest --cov=src/bvviz -v -s --ignore=tests/e2e tests/

	streamlit run app.py --server.headless true &
	pytest -k e2eTest --headed

run:
	streamlit run app.py
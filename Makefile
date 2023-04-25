lint:
	pylint --load-plugins pylint_quotes src/bvviz/
	mypy --strict --ignore-missing-imports src/bvviz/
	pylint --load-plugins pylint_quotes --disable=C0114,C0116 tests/
	mypy --strict --ignore-missing-imports --disable-error-code no-untyped-def tests/

test:
	pytest --cov=src/bvviz -v -s -n auto tests/

run:
	streamlit run app.py
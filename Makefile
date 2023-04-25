lint:
	pylint --load-plugins pylint_quotes src/bvviz/
	mypy --strict --ignore-missing-imports src/bvviz/

test:
	pytest --cov=test -v -s -n auto tests/

run:
	streamlit run app.py
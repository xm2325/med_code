.PHONY: install test demo explainable-demo

install:
	python -m pip install -e .

test:
	pytest -q

demo:
	python scripts/run_demo.py

explainable-demo:
	python scripts/run_explainable_demo.py

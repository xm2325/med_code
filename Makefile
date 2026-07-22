.PHONY: install test demo explainable-demo multilabel-demo cadec-help mimic-help

install:
	python -m pip install -e .

test:
	pytest -q

demo:
	python scripts/run_demo.py

explainable-demo:
	python scripts/run_explainable_demo.py

multilabel-demo:
	python scripts/run_multilabel_synthetic_demo.py

cadec-help:
	python scripts/run_cadec_v0012.py --help

mimic-help:
	python scripts/run_mimic_v0012.py --help

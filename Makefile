.PHONY: install install-models test demo explainable-demo multilabel-demo cadec-help cadec-model-help mimic-help

install:
	python -m pip install -e .

install-models:
	python -m pip install -e '.[models]'

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

cadec-model-help:
	python scripts/run_cadec_v0013.py --help

mimic-help:
	python scripts/run_mimic_v0012.py --help

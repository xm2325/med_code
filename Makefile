.PHONY: install test demo

install:
	python -m pip install -r requirements.txt

test:
	pytest -q

demo:
	python scripts/run_demo.py

# MedDriftGuard Makefile

.PHONY: setup test lint eval clean run-api demo

setup:
	python -m pip install --upgrade pip
	python -m pip install -e ".[dev,retrieval,ui]"

test:
	python -m pytest tests/unit -v

test-all:
	python -m pytest tests -v

lint:
	python -m ruff check src/ tests/
	python -m black --check src/ tests/

format:
	python -m ruff check --fix src/ tests/
	python -m black src/ tests/

eval:
	python eval/run_eval.py --config configs/eval.yaml

run-api:
	uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000

demo:
	python demo.py

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache build dist *.egg-info

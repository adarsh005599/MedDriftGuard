@echo off
REM Windows runner for MedDriftGuard make commands

if "%1"=="setup" (
    python -m pip install --upgrade pip
    python -m pip install -e ".[dev,retrieval,ui]"
) else if "%1"=="test" (
    python -m pytest tests/unit -v
) else if "%1"=="test-all" (
    python -m pytest tests -v
) else if "%1"=="lint" (
    python -m ruff check src/ tests/
    python -m black --check src/ tests/
) else if "%1"=="format" (
    python -m ruff check --fix src/ tests/
    python -m black src/ tests/
) else if "%1"=="eval" (
    python eval/run_eval.py --config configs/eval.yaml
) else if "%1"=="run-api" (
    uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
) else if "%1"=="demo" (
    python demo.py
) else (
    echo Usage: make.bat [setup^|test^|test-all^|lint^|format^|eval^|run-api^|demo]
)

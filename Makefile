.PHONY: test run-demo report clean

PYTHON := python3
PIP := $(PYTHON) -m pip

install:
	$(PIP) install -e ".[dev,qdrant]"

test:
	$(PYTHON) -m pytest

run-demo:
	$(PYTHON) -m vdbt run --adapters faiss,qdrant --scenarios scale_curve,noise_injection,hybrid_query,update_delete_storm,multivector_longctx --config configs/demo.json
	report:
	$(PYTHON) -m vdbt report

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf artifacts
	find . -name "__pycache__" -exec rm -rf {} +
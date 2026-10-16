.PHONY: test all demo clean install

PYTHON := python3

# ---------------------------------------------------------------------------
# Install dependencies (using pip — replace with `uv sync` if you have uv)
# ---------------------------------------------------------------------------
install:
	$(PYTHON) -m pip install -e ".[dev]" 2>/dev/null || $(PYTHON) -m pip install numpy scipy pandas matplotlib dimod dwave-neal qiskit qiskit-aer pytest

# ---------------------------------------------------------------------------
# Run tests (pytest)
# ---------------------------------------------------------------------------
test:
	$(PYTHON) -m pytest tests/ -v --tb=short

# ---------------------------------------------------------------------------
# Regenerate every result from experiments/ into results/
# ---------------------------------------------------------------------------
all: install
	$(PYTHON) experiments/run_all.py

# ---------------------------------------------------------------------------
# Demo mode — show pipeline stages and final Gantt chart (uses cached results)
# ---------------------------------------------------------------------------
demo: all
	@echo ""
	@echo "=== PIPELINE DEMO ==="
	@echo ""
	@echo "--- Results ---"
	@ls -1 results/ 2>/dev/null || echo "(no results yet — run `make all`)"
	@echo ""
	@echo "--- Sample result: small instance ---"
	@if [ -f results/annealing_small.json ]; then cat results/annealing_small.json; fi

# ---------------------------------------------------------------------------
# Cleanup generated artefacts (keeps source code)
# ---------------------------------------------------------------------------
clean:
	rm -rf results/ *.pyc __pycache__ src/__pycache__ tests/__pycache__ experiments/__pycache__

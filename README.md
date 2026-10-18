# Quantum Scheduling

This project demonstrates **job-shop scheduling**: every job consists of ordered operations, and each operation requires a specific machine for a fixed duration. The repository provides two approaches:

- a classical Google OR-Tools CP-SAT solver, used as a reference;
- a QUBO model solved with a Qiskit QAOA simulation.

The output is an operation schedule and its `makespan` — the completion time of the final operation.

> Repository status: the active implementation is in [`jobshop/`](jobshop). [`controlled code/`](controlled%20code) contains I/O models, example data, results, and the plotting utility, but its tests refer to a missing `algorithms` module. See [Known inconsistencies](#known-inconsistencies).

## Problem model

For operation `k` of job `j`, with duration `d[j, k]`, the QUBO model uses the binary variable:

```text
x[j, k, t] = 1  ⇔  operation k of job j starts at time t
```

Before creating variables, the implementation calculates the earliest and latest valid starts (`ES`, `LS`), so variables outside the `[ES, LS]` interval are pruned. The QUBO matrix includes:

1. a one-hot penalty that enforces exactly one start for each operation;
2. a penalty for violating operation precedence within a job;
3. a penalty for overlapping operations on the same machine;
4. a cost equal to the sum of the completion times of the final operations of all jobs.

The total energy is `H(x) = H_cost(x) + λ H_penalty(x)`. When `penalty_lambda` is not supplied, `jobshop.qubo_builder.calculate_sufficient_lambda()` chooses a conservative penalty weight from the cost-function coefficients.

```text
instance / InstanceData
          │
          ├── CP-SAT ────────────────► reference schedule
          │
          └── QUBO builder ─► QAOA ─► decoder ─► CalculationState
```

## Project structure

| Path | Purpose |
| --- | --- |
| [`jobshop/models.py`](jobshop/models.py) | Domain entities, built-in 2×2 and 3×3 instances, and a JSON loader. |
| [`jobshop/qubo_builder.py`](jobshop/qubo_builder.py) | Start windows, QUBO construction, and `λ` selection. |
| [`jobshop/qubo_decoder.py`](jobshop/qubo_decoder.py) | Converts QUBO bits into a `CalculationState`. |
| [`jobshop/cp_sat_solver.py`](jobshop/cp_sat_solver.py) | Exact classical OR-Tools CP-SAT solver. |
| [`jobshop/qaoa_solver.py`](jobshop/qaoa_solver.py) | QUBO-to-Ising conversion and statevector QAOA. |
| [`jobshop/compare_solvers.py`](jobshop/compare_solvers.py) | Runs and compares CP-SAT and QAOA. |
| [`controlled code/load.py`](controlled%20code/load.py), [`save.py`](controlled%20code/save.py) | Load instances and save schedules as JSON. |
| [`controlled code/plot_results.py`](controlled%20code/plot_results.py) | Generates Gantt charts from result JSON files. |
| [`controlled code/resources/`](controlled%20code/resources) | 2×2 and 3×3 instances, output schema, example schedules, and plots. |

## Requirements and installation

Python 3.11 or newer is required. Dependencies are defined in [`pyproject.toml`](pyproject.toml), including NumPy, SciPy, OR-Tools, Qiskit, Matplotlib, and the VeloxQ SDK.

The simplest setup uses `uv`:

```bash
uv sync
```

Alternatively, install the dependencies listed in `pyproject.toml` in your chosen Python environment. The project currently defines no CLI scripts or `make` targets.

## Running the solver comparison

After installing dependencies, run the comparison using the default 3×3 instance:

```bash
uv run python -m jobshop.compare_solvers
```

For the 2×2 instance:

```bash
uv run python -m jobshop.compare_solvers 2x2
```

The script:

1. obtains a CP-SAT solution;
2. builds a QUBO and reports its number of binary variables;
3. runs one-layer QAOA (`reps=1`, at most 100 COBYLA iterations);
4. decodes the bits into a schedule and prints the `makespan` and runtime comparison.

QAOA uses a full statevector and enumerates `2^N` states when calculating energies. It is therefore a demonstration for small QUBOs, rather than a method for large production instances. Use `solve_cp_sat` for an exact classical solution.

API example:

```python
from jobshop import (
    build_job_shop_qubo,
    decode_qubo_solution,
    make_test_instance,
    solve_cp_sat,
    solve_with_qaoa,
)

instance = make_test_instance("2x2")

# Classical reference solution
cp_result = solve_cp_sat(instance)
print(cp_result["makespan"])

# QUBO + QAOA path
Q, metadata = build_job_shop_qubo(instance)
result = solve_with_qaoa(Q, offset=metadata["offset"], reps=1, maxiter=100)
schedule = decode_qubo_solution(result["bits"], metadata)
print(schedule.makespan, schedule.metadata["status"])
```

## Data format

An instance JSON file contains a problem and a list of jobs with ordered operations:

```json
{
  "problem": "erp_scheduling",
  "jobs": [
    {
      "id": "Z1",
      "operations": [
        { "machine": "T", "duration": 2 },
        { "machine": "F", "duration": 1 }
      ]
    }
  ]
}
```

See [`instance_2x2.json`](controlled%20code/resources/instance_2x2.json) and [`instance_3x3.json`](controlled%20code/resources/instance_3x3.json) for complete examples. A result is saved in this form:

```json
{
  "instance": "instance_2x2",
  "method": "Method name",
  "makespan": 4,
  "schedule": [
    { "job": "Z2", "machine": "T", "start": 0, "end": 1 }
  ]
}
```

The complete schema is available in [`output_format.json`](controlled%20code/resources/output_format.json). `CalculationState.validate()` checks machine overlaps and operation counts against the instance.

## Gantt charts

Generate PNG charts from the existing result files:

```bash
uv run python "jsons/plot_results.py"
```

With no arguments, the script processes every `controlled code/resources/result_*.json` file and saves images in `controlled code/resources/plots/`. To pass a custom result file and output directory:

```bash
uv run python "jsons/plot_results.py" \
  "controlled code/resources/result_instance_2x2.json" \
  --output-dir /tmp/jobshop-plots
```

## Known inconsistencies

- `pyproject.toml` sets `testpaths = ["tests"]`, but no such directory exists in the repository.
- The tests in `controlled code/Tests/` import `algorithms.qubo`, `algorithms.sufficient_lambda`, and `algorithms.velox_solver`. The `controlled code/algorithms/` directory is absent, so these tests and `controlled code/test_velox.py` cannot run in the current tree.
- The VeloxQ integration exists only in that orphaned code and is not connected to the `jobshop/` package. `VELOX_TOKEN` would be required only after restoring that integration; do not store the token in the repository.
- [`LOG.md`](LOG.md) documents an older architecture (`anneal.py`, `bruteforce.py`, and `scheduler.py`) that is not present in the current codebase, and should not be treated as current implementation documentation.

## License

MIT — see [`LICENSE`](LICENSE).

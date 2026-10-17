# Development Log — 2026-10-17

## Job-shop scheduling implementation and project reorganisation

### Decisions made
- **Adopted a time-indexed job-shop model.** The active implementation represents an operation start with `x[j, k, t]`, using start-window pruning to avoid creating variables outside each operation's feasible earliest/latest start range.
- **Established CP-SAT as the classical reference.** Google OR-Tools CP-SAT now produces a makespan-minimising schedule with job-precedence and machine non-overlap constraints.
- **Kept QAOA as a small-instance demonstration.** The QUBO path is simulated with a full statevector, so it is appropriate for compact examples rather than production-sized instances.
- **Moved the prior prototype aside.** The previous experiment and solver code is retained in `.old/`; `jobshop/` is now the active codebase.

### Changes made
- Added the `jobshop` package:
  - domain models, JSON loading, and schedule validation;
  - QUBO construction, including one-hot, precedence, and machine-overlap penalties;
  - QUBO decoding into `CalculationState`;
  - QAOA/statevector solving and a CP-SAT/QAOA comparison runner.
- Added built-in 2×2 and 3×3 examples and made the comparison runner default to 3×3.
- Added OR-Tools, `qiskit-algorithms`, and `qiskit-optimization` to project dependencies.
- Added `controlled code/plot_results.py` and generated Gantt-chart PNGs for the supplied 2×2 and 3×3 result files.
- Added the Quantum Hackathon 2026 ERP challenge brief under `docs/`.
- Rewrote the README in English with installation, usage, architecture, data-format, visualisation, and limitation notes.

### Repository state verified from history
- The active implementation is in `jobshop/`.
- `controlled code/` remains the location for I/O helpers, example resources, saved results, and plotting.
- Historical code is preserved in `.old/` rather than deleted.

### Known issues / follow-up work
- `pyproject.toml` points pytest at `tests/`, but the repository does not currently contain that directory.
- `controlled code/Tests/` imports the former `algorithms` package, which was moved to `.old/`; those tests and the legacy VeloxQ test do not run against the current tree.
- The VeloxQ code is not connected to `jobshop/` and requires restoration before it can be used.
- The older `LOG.md` describes the pre-`jobshop` QUBO/annealing architecture. This file is the log for today's active-code transition.

### Relevant commits
- `7508c8b` — introduced the 2×2/3×3 job-shop solver path and dependencies.
- `2027802` / `7af691a` — moved the earlier implementation to `.old/`.
- `4b36a37` — added Gantt plotting and generated example charts.
- `f103de1` / `2b0f35b` — refreshed and translated the README.

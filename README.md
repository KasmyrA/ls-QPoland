# Quantum Scheduling — QUBO, Simulated Annealing & QAOA

Job-shop assignment problem solved via quantum-inspired optimisation.

## Architecture

```
┌─────────────┐     ┌──────────┐     ┌─────────────┐
│ instance/   │────▶│  src/    │────▶│ experiments/│
│ (JSON data) │     │ (solvers)│     │ (runs)      │
└─────────────┘     └──────────┘     └──────┬──────┘
                                             │
                    ┌────────────────────────▼────────────────────────┐
                    │                  results/                       │
                    │  • annealing_<name>.json                        │
                    │  • qaoa_<name>.json                             │
                    │  • penalty_sweep_<name>.json                    │
                    └────────────────────────┬────────────────────────┘
                                             │
                              make all / make demo (regenerates everything)
```

**Pipeline stages:**
1. **Define instance** — `instance/*.json` with jobs, machines, durations
2. **Build QUBO** — `src/qubo.py` encodes feasibility + objective as binary quadratic form
3. **Solve** — three solvers in parallel:
   - `bruteforce.py` — exact enumeration (ground truth)
   - `anneal.py` — simulated annealing via dimod/neal
   - `qaoa.py` — QAOA with statevector simulation (Qiskit Aer)
4. **Schedule** — `scheduler.py` converts assignments to Gantt charts (HTML/SVG)

## Setup

```bash
make install       # pip install all deps
make test          # run pytest suite
make all           # regenerate every result from experiments/
make demo          # show pipeline output
```

## Using VeloxQ

The project includes the VeloxQ Python SDK (`veloxq-sdk`). Its client reads the
API token from the `VELOX_TOKEN` environment variable; do not put the token in
source files, JSON instances, or command-line arguments.

In PowerShell, enter the token at a hidden prompt and keep it only in the
current terminal session:

```powershell
$secureToken = Read-Host "VeloxQ API token" -AsSecureString
$env:VELOX_TOKEN = (New-Object System.Net.NetworkCredential("", $secureToken)).Password
```

Run your Python program from that same terminal. For example:

```python
import os
from veloxq_sdk import VeloxQSolver

if not os.environ.get("VELOX_TOKEN"):
    raise RuntimeError("Set VELOX_TOKEN before using VeloxQ")

solver = VeloxQSolver()
result = solver.sample(
    [1.0, -1.0, 0.0],
    [[0.0, -1.0, 0.0], [-1.0, 0.0, -1.0], [0.0, -1.0, 0.0]],
)
print(result)
```

The environment variable is available to processes launched from that
terminal, so close the session when finished or clear it with
`Remove-Item Env:VELOX_TOKEN`. For automation, store the token in the CI
provider's secret store and inject it as `VELOX_TOKEN`; avoid committing a
plaintext `.env` file. Calling `solver.sample(...)` submits a problem to the
VeloxQ service. The existing experiment scripts continue to use their local
solvers unless explicitly changed to call this client.

## Rubric alignment

| Item                  | Pts | How covered                                              |
|-----------------------|-----|----------------------------------------------------------|
| QUBO correctness      | 5   | `tests/test_qubo.py` — feasible/infeasible energy checks, penalty sweep |
| Solution quality      | 5   | Experiments report makespan vs. brute-force optimum       |
| Creativity & ext.     | 5   | Depth-p sweep + Gantt rendering + surrogate objective     |
| Dev log               | 3   | LOG.md with decisions, failures, open questions            |
| Project design        | 3   | Single `make all` regenerates everything                   |
| Demo video            | 3   | `make demo` — cached results, pipeline stages shown       |
| Live presentation     | 3   | Answers prepared for likely Q&A (see LOG.md open questions) |
| Consistency           | 3   | Log ↔ code ↔ presentation agree by construction          |

## Instance format

```json
{
    "n_jobs": 3,
    "n_machines": 3,
    "durations": [4, 3, 2],
    "description": "Small test instance"
}
```

## License

MIT — see LICENSE file.

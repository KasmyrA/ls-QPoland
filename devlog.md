# Development Log

This log describes the active implementation in `qubo/`, `jsons/`, and
`tests/results_runner.py`. The older `LOG.md` records an earlier prototype and
is not the source of truth for the current solver.

## Current implementation and decisions

### Time-indexed operation choices

The model uses a binary variable for each feasible combination of operation,
physical machine unit, and integer start time:

```text
x[j,k,u,t] = 1  iff operation k of job j starts on unit u at time t
```

This was chosen over an assignment-only encoding because this project must
represent *when* each operation runs: job order, machine conflicts, alternate
units, outages, and duty-cycle limits all depend on times and durations. A
time-indexed model makes those rules explicit as linear or pairwise quadratic
terms. Its main cost is variable growth with the time horizon, so start windows
and blocked-time pruning are applied before variables are created.

The builder computes earliest starts using minimum possible durations of
earlier operations and latest starts using the horizon, due-end bounds, and
minimum durations of later operations. It then removes starts that overlap
outages or frozen work. The solver tries bounded candidate horizons, beginning
near a greedy schedule/lower bound; it accepts the first horizon at which its
sampler returns a feasible schedule. For simulated annealing and quantum
sampling this is a practical search strategy, not a proof of global optimality.

### Hard rules become quadratic penalties

All hard constraints are represented in the QUBO rather than left to the
sampler as external constraints:

- **Exactly one choice per operation:** a squared one-hot penalty across that
  operation's feasible unit/start candidates.
- **Job precedence:** a pairwise penalty for every pair of consecutive
  operation choices where the second starts before the first finishes.
- **Unit capacity:** a pairwise penalty for overlapping choices of two
  different operations assigned to the same physical unit.
- **Duty cycle:** a squared equality using integer slack bits for each modeled
  unit/time window.
- **Makespan linkage:** when minimizing makespan, one-hot time variables encode
  the selected makespan and pairwise penalties forbid a selected makespan
  earlier than any job's final-operation completion.

These terms are all non-negative when interpreted as constraint violations.
The same global penalty multiplier is used for them. This keeps the formulation
consistent and lets the QUBO be sent to samplers that accept only an objective
function.

### Penalty-weight selection

The active default is `lambda_mode="tight"`, unless a caller supplies an
explicit `penalty_lambda`. The builder sets:

```text
lambda_tight = max(1, 2 * max_i |Q_cost[i,i]|)
```

This practical heuristic scales constraint penalties against the largest
diagonal objective coefficient, with a factor of two to discourage constraint
violations without making the energy landscape unnecessarily steep for
annealing. It is not a universal proof that every infeasible state is more
expensive than every feasible state; returned plans are therefore decoded and
validated against the original constraints.

The alternative `lambda_mode="sufficient"` uses a conservative coefficient
bound:

```text
lambda_sufficient =
  (sum_i |Q_cost[i,i]| + sum_{i<j} |Q_cost[i,j]|) / delta + safety_margin
```

The implementation uses `delta=1` and a safety margin of `1` by default. The
bound is intended to make the penalty dominate the possible objective change
from a binary assignment; it can be much larger than the practical setting and
make heuristic search harder. The results runner uses this mode for its
optional VeloxQ run, while its local SA and QAOA paths use the builder's tight
default. The documented formula is for the symmetric QUBO cost matrix, as
constructed here.

### Duty-cycle slack encoding and annealing

For a limit of at most `X` work units in any window of length `X+Y`, the model
adds slack `z` and squares the difference between assigned work plus slack and
the available work budget. Binary-weighted slack is the default: it uses
logarithmically many bits (with a final adjusted weight to represent every
integer through `X`). Unary slack is available as an alternative; it uses more
variables but allows one-bit changes to move the encoded slack by one unit.
Binary was selected to keep the QUBO smaller.

Single-bit annealing moves often cannot adjust binary slack and the schedule
bits together. The local SA sampler therefore has a structured mode that
analytically eliminates duty slack during search and reconstructs valid slack
bits for each returned schedule. This is an optimization of the sampler, not a
different model: returned energies are evaluated using the full QUBO. Other
samplers receive the ordinary QUBO, including its slack variables.

### Solver roles and run settings

- **Simulated annealing (SA):** local QUBO sampler usable without a cloud
  service. The runner uses 48 reads, 300 sweeps, and seed 0. The general
  pipeline defaults are 96 reads and 600 sweeps.
- **CP-SAT:** time-indexed classical reference solver and feasibility
  cross-check. The results runner allows 60 seconds and uses eight workers.
- **QAOA:** small-instance quantum-algorithm demonstration. The results runner
  uses a statevector simulator with 2 layers, up to 150 COBYLA iterations,
  returns up to 256 high-probability states, and caps the model at 16 qubits.
  Statevector simulation scales exponentially, so this is not the large
  instance backend.
- **VeloxQ:** optional cloud sampler. Its run requires the external SDK and a
  token; credentials are not stored in the repository.
- **Rolling horizon:** bounds the subproblem by the next two operations per
  job, commits operations that start before the next three-time-unit step, then
  advances time. This controls subproblem size at the cost of making decisions
  before the whole schedule is optimized.

### Breakdowns and rescheduling

The rescheduler preserves completed work and work in progress on healthy
units, blocks outage intervals, and replans the remaining operations. When an
operation is interrupted by a breakdown, its work before the interruption is
counted as committed machine workload for duty-cycle accounting and the
operation is scheduled again at its full duration. If replanning fails, the
pipeline falls back to a right-shift schedule that keeps the prior unit/order
where possible.

## Recorded result snapshots

These are observations from files currently present in `results/`; they are
not claims that every solver setting or penalty value is recorded in each
result file.

| Case | Recorded result |
| --- | --- |
| 3×3 static plan | SA matched the CP-SAT reference makespan of 7 with 53 QUBO variables and zero validation violations. See `results/base3x3/3x3_base__plan__sa.meta.json`. |
| 3×3 rolling plan | Both SA and CP-SAT recorded cost 8 against the one-shot reference cost 7 (14.3% gap), with zero violations. This illustrates the quality trade-off of committing decisions in slices. See `results/base3x3/3x3_base__rolling__sa.meta.json`. |
| 2×2 QAOA | The recorded statevector run found cost 4, matching its CP-SAT reference, using 10 variables. See `results/quantum/q2x2_basic__plan__qaoa.meta.json`. |
| 6×3 duty-cycle plan | SA recorded cost 11, matching the reference, with 276 variables and zero violations. See `results/real/6x3_duty__plan__sa.meta.json`. |
| 3×3 VeloxQ attempts | The saved summary records no feasible static sample at 133 variables and an error in rolling mode. These runs are evidence of current backend/size limitations, not a comparison proving that the QUBO itself is infeasible. See `results/summary.csv`. |

The checked-in summary snapshot currently contains only the base 3×3 rows;
individual result files and metadata exist for other cases as well. Metrics are
specific to the saved runs and should be regenerated before making broader
performance claims.

## Current limitations to keep visible

- Time is discrete and durations are integer-valued.
- SA and statevector QAOA are heuristic samplers; a feasible sample is not an
  optimality certificate.
- Rolling horizon is approximate and can have a worse objective than a
  one-shot solve.
- Result metadata does not persist every builder parameter (notably lambda and
  slack encoding), so those values should not be inferred from a result cost
  or variable count alone.

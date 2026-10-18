# Project Design: Time-Indexed QUBO Scheduling

## 1. Problem represented

The project schedules a set of jobs on machine units. Each job is a chain of
ordered operations. An operation takes a fixed positive integer duration on an
eligible machine unit; where machine configuration supplies alternatives, the
duration may depend on the selected unit. A schedule chooses a unit and start
time for every operation.

This is a deterministic, discrete-time, non-preemptive job-shop/ERP scheduling
model. For operation `k` of job `j`, define its selected start `s[j,k]`,
selected unit `u[j,k]`, and processing time `p[j,k,u]`. Its completion time is

```text
C[j,k] = s[j,k] + p[j,k,u[j,k]]
```

Two supported objectives are:

```text
Makespan:          minimize C_max = max_j C[j,last]
Sum of completions: minimize sum_j C[j,last]
```

The objective is measured at each job's final operation. There are no
weights, tardiness costs, setup costs, or objectives on intermediate
operations in this implementation.

## 2. Decision-variable encoding

For every feasible operation/unit/integer-start combination, the builder
creates:

```text
x[j,k,u,t] ∈ {0,1}
x[j,k,u,t] = 1  iff operation k of job j starts on unit u at time t
```

Only eligible combinations are generated. The time horizon, earliest/latest
start bounds, release and due-end bounds, and blocked intervals prune the
candidate set. Machine configurations expand machine types into physical
units; duration overrides can add alternative unit/type choices. An operation
must still select exactly one of its remaining candidates.

For the makespan objective, auxiliary binary variables encode the selected
makespan:

```text
y[T] ∈ {0,1}
y[T] = 1  iff the encoded makespan is T
```

For duty-cycle constraints, auxiliary binary slack variables `z[w,b]` encode
an integer amount of unused work capacity in window `w`. These are encoding
variables, not additional schedule decisions.

## 3. Constraints

### 3.1 Exactly one unit/start for every operation

For each operation, exactly one candidate must be selected:

```text
sum_(u,t) x[j,k,u,t] = 1
P_one(j,k) = (sum_(u,t) x[j,k,u,t] - 1)^2
```

This is a local/separable constraint: its one-hot group is specific to one
operation. It does not by itself decide whether the resulting choices satisfy
precedence or machine capacity.

### 3.2 Job precedence — dependent across operations

Every operation must finish before the next operation in the same job starts:

```text
s[j,k+1] >= s[j,k] + p[j,k,u[j,k]]
```

Instead of introducing a higher-order condition, the QUBO adds a quadratic
term for every pair of candidate starts that violates this rule:

```text
P_prec = sum x[j,k,u,t] * x[j,k+1,v,t']
         over pairs where t' < t + p[j,k,u]
```

Thus each penalty couples consecutive operations in a job, independently of
which units those operations use.

### 3.3 Unit capacity — dependent across operations using the same unit

Distinct operations cannot occupy the same physical unit at overlapping
times. Intervals are half-open, `[t, t+p)`, so an operation ending at time `t`
does not conflict with another starting at `t`.

For two candidate intervals on the same unit, overlap is detected by:

```text
max(t1,t2) < min(t1+p1,t2+p2)
```

The QUBO adds `x1*x2` for each overlapping pair from different operations.
Capacity constraints for different physical units are separate; job
precedence can still connect operations scheduled on different units.

### 3.4 Duty cycle — dependent across all work in a window

If unit `u` has configuration `(X[u],Y[u])`, define window length
`L[u]=X[u]+Y[u]`. In every modeled window of that length, the total duration
of work occupying the window must not exceed `X[u]`.

The contribution of candidate operation `i` to a window starting at `a` is its
overlap duration:

```text
c[i,u,a] = overlap([t_i,t_i+p_i), [a,a+L[u]))
W[u,a]   = sum_i c[i,u,a] * x[i]
```

Already committed work (for example, in breakdown rescheduling) consumes part
of the budget. The remaining capacity is:

```text
X_eff[u,a] = max(0, X[u] - sum_committed overlap(work, window))
```

Slack weights `a[b]` encode an integer from 0 to `X_eff`. The constraint and
its QUBO penalty are:

```text
W[u,a] + sum_b a[b] * z[u,a,b] = X_eff[u,a]
P_duty(u,a) = (W[u,a] + sum_b a[b]*z[u,a,b] - X_eff[u,a])^2
```

The default binary weights use logarithmically many bits, with an adjusted
last weight so every integer slack value up to the capacity is representable.
Unary encoding is available but uses one bit per capacity unit. Windows whose
maximum possible work cannot exceed their available capacity are omitted.

This is a cumulative-window rule: it limits total work in every modeled
`X+Y`-length window. It does not simulate a separate machine-rest state or
explicitly track a cooling process.

### 3.5 Outside-world restrictions

The `Fixed` input supports:

- earliest start (`release`) and latest end (`due_end`) bounds;
- machine outages;
- committed/frozen machine work.

These are mostly enforced structurally: impossible candidate starts are
removed before building the QUBO. Committed work also reduces duty-cycle
capacity in affected windows. Such restrictions do not need separate
decision variables.

## 4. QUBO objective and penalty construction

The complete energy is:

```text
E(q) = qᵀ Q q + offset
     = C(q) + λ * P(q)
```

where `q` contains schedule, makespan, and duty-slack bits; `C` is the chosen
objective; and `P` is the sum of one-hot, precedence, unit-overlap,
duty-cycle, and (for makespan) makespan-link penalties. The implementation
stores a symmetric matrix and divides pair coefficients between its two
entries, so a quadratic interaction appears once in `qᵀQq`. Constant terms
from squared penalties are kept in `offset`.

### Makespan QUBO

The max function is not directly linear in the operation-start bits. The
model introduces one-hot `y[T]` values, minimizes the selected `T`, and
forbids a makespan value below any selected final-operation completion:

```text
C_makespan(q) = sum_T T*y[T]
P_y_one       = (sum_T y[T] - 1)^2
P_link        = sum_(j,t,u,T : t+p[j,last,u] > T)
                x[j,last,u,t] * y[T]
```

At a feasible minimum, the selected `T` is the smallest value that is at
least every job's completion time, hence the makespan.

### Sum-of-completions QUBO

Completion time for a selected final-operation candidate is linear in its
selection bit, so this alternative needs no makespan variables:

```text
C_sum(q) = sum_j sum_(u,t) (t + p[j,last,u]) * x[j,last,u,t]
```

Both objective terms are diagonal/linear QUBO contributions. The quadratic
terms primarily represent interactions required by constraints.

## 5. Penalty multiplier `lambda`

Unless explicitly supplied, the QUBO builder uses the tight heuristic:

```text
λ_tight = max(1, 2 * max_i |Q_cost[i,i]|)
```

This scales the penalty against the largest diagonal objective coefficient
and aims to balance constraint enforcement with a less steep energy landscape
for heuristic samplers. It is a practical setting, not a general optimality
or feasibility guarantee.

The alternative sufficient setting uses a coefficient-magnitude bound over
the symmetric objective QUBO:

```text
λ_sufficient =
  (sum_i |Q_cost[i,i]| + sum_(i<j) |Q_cost[i,j]|) / δ + 1
```

The code defaults to `δ=1`, the smallest nonzero squared-integer violation,
and safety margin `1`. This conservative choice is intended to make a
constraint violation too expensive to offset by changing the objective; it
can produce much larger coefficients. Explicit `penalty_lambda` overrides
either calculation. The objective and penalty functions include constants;
the constants shift energy values but do not change which assignment
minimizes the QUBO.

Every returned candidate is decoded and checked against the schedule
constraints. Feasibility validation is important with heuristic samples and
with the tight lambda mode.

## 6. Rolling horizon and rescheduling

### Rolling horizon

Rolling scheduling solves a smaller subproblem containing at most the next
`lookahead` operations of each unfinished job. With the runner defaults,
`lookahead=2` operations and `step=3` time units. It commits operations whose
starts fall before the next step boundary and then advances the current time.
This reduces the size of each optimization but the decisions are local: later
operations are not all optimized jointly, so rolling schedules can be worse
than one-shot schedules. The saved 3×3 results show cost 8 for rolling SA and
CP-SAT versus 7 for the one-shot reference.

### Breakdown rescheduling

Completed work and healthy work already in progress can be frozen. Operations
affected by an outage are replanned; work completed before an interruption is
retained as committed machine workload for duty-cycle accounting, and the
interrupted operation is restarted at its full duration. If constrained
replanning does not yield a valid plan, the implementation falls back to a
right-shift baseline.

## 7. Solver implementation and scale

- **SA** searches the QUBO by single-bit flips; the runner uses 48 parallel
  reads, 300 sweeps, and seed 0. Structured SA removes duty-slack bits during
  the search and reconstructs them exactly afterward.
- **Exact QUBO enumeration** is available for very small models (default
  maximum 24 variables), but grows as `2^n`.
- **QAOA** converts the QUBO to an Ising Hamiltonian and uses a Qiskit
  statevector. The results runner uses 2 QAOA layers, COBYLA with up to 150
  iterations, and a 16-qubit maximum; statevector work also scales
  exponentially.
- **CP-SAT** provides a separate time-indexed classical reference model with
  matching precedence, machine, duty-cycle, and fixed-work semantics. It is
  used to compare objective values and validate the QUBO path.
- **VeloxQ** is an optional cloud backend; it receives the plain slack QUBO.

The results runner uses a 60-second CP-SAT time limit and eight workers. These
are experiment settings, not properties of the scheduling problem.

## 8. Assumptions and simplifications

1. **Discrete, deterministic time.** Starts, durations, outage boundaries, and
   committed intervals are integer time units; processing durations are fixed
   for a selected unit.
2. **Non-preemptive operations in a plan.** Each selected operation occupies
   one continuous interval. Breakdown rescheduling restarts an interrupted
   operation rather than modeling remaining processing time.
3. **Fixed job routes.** Job operations have a fixed order. There are no
   precedence alternatives, operation splitting, or transport operations.
4. **No setup/sequence costs.** The model does not represent setup times,
   sequence-dependent transitions, machine changeover penalties, labor,
   inventory, or energy cost.
5. **Machine capacity is unary per physical unit.** Parallel identical
   machines are expanded into individual units; each unit handles at most one
   operation at a time.
6. **Duty cycle is a sliding-window workload cap.** It uses total overlap in
   each `X+Y` window. It does not model thermal state, calendar exceptions, or
   a distinct mandatory-rest transition.
7. **Makespan and final completion only.** The built-in objectives are
   makespan or unweighted sum of final completion times; release/due-end
   bounds are supplied through fixed constraints, not optimized tardiness
   penalties.
8. **Finite horizon.** Only starts inside the selected finite horizon are
   represented. Horizon search and candidate pruning control model size but
   cannot prove optimality when the sampler is heuristic or only a bounded
   subset of horizons is tried.
9. **Sampler limits matter.** A low-energy or feasible sample is not by itself
   an optimality certificate. The pipeline validates schedules and CP-SAT
   provides an independent reference where it can solve the instance.

## 9. Recorded examples

The saved artifacts illustrate model behavior, not universal performance:

- `results/base3x3/3x3_base__plan__sa.meta.json`: SA cost 7, reference 7,
  53 variables, zero violations.
- `results/base3x3/3x3_base__rolling__sa.meta.json`: rolling SA cost 8,
  reference 7, zero violations.
- `results/quantum/q2x2_basic__plan__qaoa.meta.json`: QAOA cost 4, reference
  4, 10 variables.
- `results/real/6x3_duty__plan__sa.meta.json`: duty-cycle SA cost 11,
  reference 11, 276 variables, zero violations.

The checked-in `results/summary.csv` is currently a partial snapshot; inspect
the individual case metadata before drawing conclusions across all instances.

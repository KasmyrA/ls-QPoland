"""QUBO builder for the job-assignment problem.

Encoding (n_jobs x n_machines = N bits):
    bit b[j*m + m_idx]  == 1  ⇔  job j is assigned to machine m_idx.

Feasibility: every row must sum to exactly 1.
Objective:   minimise makespan ≈ max over machines of (sum of durations on that machine).

Because pure QUBO cannot directly express a max, we use two penalties and
a surrogate objective (total completion time) which is tightly correlated.
"""
from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bit_index(j: int, m: int, n_machines: int) -> int:
    return j * n_machines + m


def _row_penalty_matrix(n_jobs: int, n_machines: int) -> tuple[np.ndarray, np.ndarray]:
    """Build QUBO rows for the constraint Σ_m x[j,m]=1 for each job j.

    Returns (Q matrix, linear offset).  The penalty term is::

        A * sum_j (sum_m x[j,m] - 1)^2
          = A * [ sum_j sum_m x[j,m]^2 + 2*sum_j sum_{m<m'} x[j,m]*x[j,m']
                  - 2*sum_j sum_m x[j,m] + n_jobs ]

    Since x^2 = x for binary variables, the diagonal picks up A*n_jobs extra.
    """
    N = n_jobs * n_machines
    Q = np.zeros((N, N))

    diag_shift = 0.0  # constant offset from expanding (Σx - 1)^2
    for j in range(n_jobs):
        bits = [_bit_index(j, m, n_machines) for m in range(n_machines)]
        # diagonal: each x[j,m]^2 == x[j,m] → contributes A * (1 - 2) = -A per bit
        for b in bits:
            Q[b, b] -= 1.0

        # off-diagonal: every pair gets +2A
        for i_idx, a in enumerate(bits):
            for b in bits[i_idx + 1 :]:
                Q[a, b] += 2.0
                Q[b, a] += 2.0

        diag_shift += 1.0  # one constant term per row expansion

    return Q, diag_shift


def _surrogate_objective_matrix(
    durations: np.ndarray, n_machines: int
) -> tuple[np.ndarray, float]:
    """Build QUBO terms for total completion time (surrogate for makespan).

    For each machine m the load is L_m = Σ_j d[j] * x[j,m].
    Total cost = Σ_m L_m^2  (expands to diagonal + pairwise within-machine).
    """
    n_jobs = durations.shape[0]
    N = n_jobs * n_machines
    Q = np.zeros((N, N))

    for m in range(n_machines):
        bits = [_bit_index(j, m, n_machines) for j in range(n_jobs)]
        # Expand (Σ_j d[j] x[j,m])^2:
        #   diagonal: d[j]^2 on bit (j,m)
        #   off-diag  : 2*d[j]*d[k] between bits (j,m),(k,m)
        for a in bits:
            Q[a, a] += durations[a // n_machines] ** 2

        for j_a_idx, a in enumerate(bits):
            for b in bits[j_a_idx + 1 :]:
                d_a = durations[a // n_machines]
                d_b = durations[b // n_machines]
                term = 2.0 * d_a * d_b
                Q[a, b] += term
                Q[b, a] += term

    return Q, 0.0


def _bit_to_job(bit: int, n_machines: int) -> int:
    return bit // n_machines


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_qubo(
    durations: np.ndarray,
    penalty_weight: float | None = None,
    objective_weight: float = 1.0,
) -> dict:
    """Assemble the full QUBO and return it as a dict consumable by dimod.

    Parameters
    ----------
    durations : (n_jobs,) array of processing times.
    penalty_weight : A — weight of the feasibility constraint. Defaults to
        max(durations)^2 * n_machines * 10, which is sufficient to dominate
        any surrogate-objective contribution for square instances.
    objective_weight : B — weight of the surrogate objective.
    """
    n_jobs = durations.shape[0]
    n_machines = n_jobs  # square instance

    if penalty_weight is None:
        penalty_weight = float(np.max(durations) ** 2 * n_machines * 10)

    Q_const, diag_off = _row_penalty_matrix(n_jobs, n_machines)
    Q_obj, obj_off = _surrogate_objective_matrix(durations, n_machines)

    Q = penalty_weight * Q_const + objective_weight * Q_obj
    offset = penalty_weight * diag_off + objective_weight * obj_off

    # dimod expects {(u, v): bias} format where u <= v are variable indices
    qubo_pairs: dict[tuple[int, int], float] = {}
    for i in range(Q.shape[0]):
        for j in range(i, Q.shape[1]):
            if abs(Q[i, j]) > 1e-12:
                qubo_pairs[(i, j)] = float(Q[i, j])

    return {
        "Q": Q,
        "offset": offset,
        "qubo": qubo_pairs,
        "penalty_weight": penalty_weight,
        "objective_weight": objective_weight,
        "n_bits": n_jobs * n_machines,
        "durations": durations,
    }


def decode(bitstring: np.ndarray, n_machines: int) -> list[int]:
    """Convert a flat 1-D binary vector to a job→machine assignment.

    Returns a length `n_jobs` list of machine indices (one per job).
    Infeasible rows are flagged as -1.
    """
    n_jobs = len(bitstring) // n_machines
    assignment: list[int] = []
    for j in range(n_jobs):
        row = bitstring[j * n_machines : (j + 1) * n_machines]
        s = int(row.sum())
        if s == 0:
            assignment.append(-1)  # unassigned
        elif s > 1:
            assignment.append(j)   # multi-assigned (ambiguous — pick first)
        else:
            assignment.append(int(np.argmax(row)))
    return assignment


def energy(assignment_flat: np.ndarray, n_machines: int, qubo: dict) -> float:
    """Compute the QUBO energy of a flat binary bitstring."""
    N = len(assignment_flat)
    bits = assignment_flat.astype(int)
    e = 0.0
    for i in range(N):
        if bits[i]:
            e += qubo["Q"][i, i] * bits[i]
            for j in range(i + 1, N):
                if bits[j]:
                    e += 2.0 * qubo["Q"][i, j] * bits[i] * bits[j]
    return float(e) + qubo["offset"]


# ---------------------------------------------------------------------------
# Penalty sweep utility (for tests / experiments)
# ---------------------------------------------------------------------------

def penalty_sweep(durations: np.ndarray, weights=(1.0, 5.0, 10.0, 50.0, 200.0)) -> dict:
    """Run build_qubo across several penalty_weight values."""
    results = []
    for w in weights:
        q = build_qubo(durations, penalty_weight=w)
        results.append({
            "penalty_weight": w,
            "n_feasible_rows": 0,  # filled by caller if desired
            "Q_norm": float(np.linalg.norm(q["Q"], ord="fro")),
        })
    return {"sweep": results}

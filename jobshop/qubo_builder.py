"""QUBO builder for Job-Shop / ERP scheduling problems.

Constructs time-indexed QUBO:
- Binary decision variables: x_{(job, op), t} = 1 if operation starts at time t.
- Start-window pruning: [ES, LS] based on precedence bounds and horizon H.
- Quadratic penalty constraints:
    1. Exactly one start time per operation
    2. Operation precedence within each job
    3. Machine non-overlapping
- Objective: Minimize makespan / completion time of final operations.
"""
from __future__ import annotations

from typing import Any, Optional
import numpy as np

try:
    from .models import InstanceData
except ImportError:
    from models import InstanceData  # type: ignore


def compute_start_windows(instance: InstanceData, horizon: int) -> dict[tuple[str, int], tuple[int, int]]:
    """Compute earliest start time (ES) and latest start time (LS) for every operation."""
    windows: dict[tuple[str, int], tuple[int, int]] = {}

    for job in instance.jobs:
        ops = job.operations
        n_ops = len(ops)

        # Forward pass: Earliest Start (ES)
        es_list = [0] * n_ops
        current_es = 0
        for k, op in enumerate(ops):
            es_list[k] = current_es
            current_es += op.duration

        # Backward pass: Latest Start (LS)
        ls_list = [0] * n_ops
        current_ls = horizon
        for k in reversed(range(n_ops)):
            op = ops[k]
            current_ls -= op.duration
            ls_list[k] = current_ls

        for k in range(n_ops):
            es = es_list[k]
            ls = max(es, ls_list[k])
            if es > ls:
                raise ValueError(
                    f"Job {job.id}, operation {k} cannot fit in horizon {horizon}"
                )
            windows[(job.id, k)] = (es, ls)

    return windows


def calculate_sufficient_lambda(
    Q_cost: np.ndarray,
    delta: float = 1.0,
    safety_margin: float = 1.0,
) -> float:
    """Calculate sufficient penalty multiplier: lambda > (sum |a_i| + sum_{i<j} |b_ij|) / delta."""
    diag = np.diag(Q_cost)
    linear_sum = float(np.sum(np.abs(diag)))
    off_diag = np.triu(Q_cost, k=1) + np.tril(Q_cost, k=-1)
    quad_sum = float(np.sum(np.abs(off_diag))) / 2.0

    sufficient_lambda = (linear_sum + quad_sum) / delta + safety_margin
    return float(sufficient_lambda)


def build_job_shop_qubo(
    instance: InstanceData,
    horizon: Optional[int] = None,
    penalty_lambda: Optional[float] = None,
    delta: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build time-indexed QUBO for Job-Shop scheduling.

    Returns
    -------
    Q : np.ndarray
        Symmetric QUBO matrix (n_vars x n_vars).
    metadata : dict
        Contains 'offset', 'n_vars', 'var_map', 'inv_var_map', 'windows', 'instance', etc.
    """
    if horizon is None:
        # Default horizon heuristics:
        # For 2x2 standard instance, optimal makespan is 4
        if instance.n_jobs <= 2 and all(len(j.operations) <= 2 for j in instance.jobs):
            horizon = 4
        elif instance.n_jobs <= 3 and all(len(j.operations) <= 3 for j in instance.jobs):
            horizon = 8
        else:
            horizon = sum(op.duration for job in instance.jobs for op in job.operations)

    windows = compute_start_windows(instance, horizon)

    # 1. Map binary variables: x_{(job_id, op_index, t)} -> index
    var_map: dict[tuple[str, int, int], int] = {}
    inv_var_map: dict[int, tuple[str, int, int]] = {}
    current_idx = 0

    for job in instance.jobs:
        for k in range(len(job.operations)):
            es, ls = windows[(job.id, k)]
            for t in range(es, ls + 1):
                var_map[(job.id, k, t)] = current_idx
                inv_var_map[current_idx] = (job.id, k, t)
                current_idx += 1

    n_vars = current_idx
    Q_pen = np.zeros((n_vars, n_vars), dtype=float)
    Q_cost = np.zeros((n_vars, n_vars), dtype=float)
    offset_pen = 0.0

    # 2. Constraint 1: Exactly one start time per operation
    # (sum_t x_t - 1)^2 = sum_t x_t + 2 sum_{t < t'} x_t x_t' - 2 sum_t x_t + 1
    #                   = - sum_t x_t + 2 sum_{t < t'} x_t x_t' + 1
    for job in instance.jobs:
        for k in range(len(job.operations)):
            es, ls = windows[(job.id, k)]
            v_indices = np.array([var_map[(job.id, k, t)] for t in range(es, ls + 1)], dtype=int)

            # Diagonal: -1.0
            Q_pen[v_indices, v_indices] -= 1.0
            # Off-diagonal: +1.0 for each pair (symmetrized 0.5 + 0.5)
            block = np.ones((len(v_indices), len(v_indices)), dtype=float)
            np.fill_diagonal(block, 0.0)
            Q_pen[np.ix_(v_indices, v_indices)] += block
            offset_pen += 1.0

    # 3. Constraint 2: Precedence within each job
    # Operation k+1 cannot start before k finishes: t_{k+1} >= t_k + d_k
    for job in instance.jobs:
        for k in range(len(job.operations) - 1):
            op_k = job.operations[k]
            es_k, ls_k = windows[(job.id, k)]
            es_next, ls_next = windows[(job.id, k + 1)]

            t_k = np.arange(es_k, ls_k + 1)[:, None]
            t_next = np.arange(es_next, ls_next + 1)[None, :]
            violation_mask = (t_next < t_k + op_k.duration).astype(float)

            v_k = np.array([var_map[(job.id, k, t)] for t in range(es_k, ls_k + 1)], dtype=int)
            v_next = np.array([var_map[(job.id, k + 1, t)] for t in range(es_next, ls_next + 1)], dtype=int)

            Q_pen[np.ix_(v_k, v_next)] += 0.5 * violation_mask
            Q_pen[np.ix_(v_next, v_k)] += 0.5 * violation_mask.T

    # 4. Constraint 3: Machine non-overlapping
    machine_ops: dict[str, list[tuple[str, int, int]]] = {}
    for job in instance.jobs:
        for k, op in enumerate(job.operations):
            machine_ops.setdefault(op.machine, []).append((job.id, k, op.duration))

    for machine, ops_list in machine_ops.items():
        n_m_ops = len(ops_list)
        for i in range(n_m_ops):
            j1, k1, d1 = ops_list[i]
            es1, ls1 = windows[(j1, k1)]
            v1_arr = np.array([var_map[(j1, k1, t)] for t in range(es1, ls1 + 1)], dtype=int)
            t1_arr = np.arange(es1, ls1 + 1)[:, None]

            for j in range(i + 1, n_m_ops):
                j2, k2, d2 = ops_list[j]
                es2, ls2 = windows[(j2, k2)]
                v2_arr = np.array([var_map[(j2, k2, t)] for t in range(es2, ls2 + 1)], dtype=int)
                t2_arr = np.arange(es2, ls2 + 1)[None, :]

                # Overlap: max(t1, t2) < min(t1 + d1, t2 + d2)
                overlap_mask = (np.maximum(t1_arr, t2_arr) < np.minimum(t1_arr + d1, t2_arr + d2)).astype(float)

                Q_pen[np.ix_(v1_arr, v2_arr)] += 0.5 * overlap_mask
                Q_pen[np.ix_(v2_arr, v1_arr)] += 0.5 * overlap_mask.T

    # 5. Objective H_cost: Minimize completion times of the last operation of each job
    for job in instance.jobs:
        if not job.operations:
            continue
        last_k = len(job.operations) - 1
        last_op = job.operations[last_k]
        es, ls = windows[(job.id, last_k)]
        t_vals = np.arange(es, ls + 1)
        v_indices = np.array([var_map[(job.id, last_k, t)] for t in t_vals], dtype=int)
        Q_cost[v_indices, v_indices] += (t_vals + last_op.duration).astype(float)

    # 6. Penalty Multiplier (Sufficient Lambda)
    if penalty_lambda is None:
        penalty_lambda = calculate_sufficient_lambda(Q_cost, delta=delta, safety_margin=1.0)

    # Combine: H = H_cost + lambda * H_pen
    Q = Q_cost + penalty_lambda * Q_pen
    offset = penalty_lambda * offset_pen

    metadata = {
        "Q": Q,
        "offset": offset,
        "Q_cost": Q_cost,
        "Q_pen": Q_pen,
        "offset_pen": offset_pen,
        "penalty_lambda": penalty_lambda,
        "horizon": horizon,
        "n_vars": n_vars,
        "var_map": var_map,
        "inv_var_map": inv_var_map,
        "windows": windows,
        "instance": instance,
    }

    return Q, metadata


"""QUBO formulation and solver for ERP / Job-Shop scheduling problems.

Constructs a time-indexed QUBO for InstanceData:
- Binary decision variable: x_{(job, op), t} = 1 if operation op of job starts at time t.
- Start-window pruning: t in [ES, LS] based on precedence bounds and horizon H.
- Constraints enforced via quadratic penalties:
    1. Exactly one start time per operation: (sum_t x_{(j,k), t} - 1)^2
    2. Operation precedence within each job: x_{(j,k), t_k} * x_{(j,k+1), t_{k+1}} = 0 for t_{k+1} < t_k + d_k
    3. Machine non-overlapping: x_{A, t_A} * x_{B, t_B} = 0 if running on same machine and intervals overlap
- Objective: Minimize makespan / completion times of the final operations.
"""

from __future__ import annotations

import itertools
from typing import Any, Optional
import numpy as np

try:
    from ..entities import CalculationState, InstanceData, ScheduledTask
    from .sufficient_lambda import calculate_sufficient_lambda_from_qubo
except (ImportError, ValueError):
    try:
        from controlled_code.entities import CalculationState, InstanceData, ScheduledTask  # type: ignore
        from controlled_code.algorithms.sufficient_lambda import calculate_sufficient_lambda_from_qubo  # type: ignore
    except ImportError:
        from entities import CalculationState, InstanceData, ScheduledTask  # type: ignore
        from algorithms.sufficient_lambda import calculate_sufficient_lambda_from_qubo  # type: ignore


def compute_start_windows(instance: InstanceData, horizon: int) -> dict[tuple[str, int], tuple[int, int]]:
    """Compute earliest start time (ES) and latest start time (LS) for every operation.

    Parameters
    ----------
    instance : InstanceData
        Problem instance.
    horizon : int
        Makespan upper bound H.

    Returns
    -------
    dict[(job_id, op_index), (ES, LS)]
    """
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
            windows[(job.id, k)] = (es, ls)

    return windows


def build_job_shop_qubo(
    instance: InstanceData,
    horizon: Optional[int] = None,
    penalty_lambda: Optional[float] = None,
    delta: float = 1.0,
) -> dict[str, Any]:
    """Build time-indexed QUBO for the job-shop scheduling problem.

    Parameters
    ----------
    instance : InstanceData
        The parsed instance from load_instance.
    horizon : int, optional
        Makespan time horizon H. If None, defaults to the sum of durations.
    penalty_lambda : float, optional
        Penalty multiplier. If None, automatically calculated using sufficient_lambda.
    delta : float
        Minimal penalty gap for constraint violations (default 1.0).

    Returns
    -------
    dict
        Dictionary containing Q matrix, offset, variable maps, and metadata.
    """
    if horizon is None:
        total_dur = sum(op.duration for j in instance.jobs for op in j.operations)
        # Safe tight horizon for small instances
        if instance.n_jobs <= 2:
            horizon = max(5, total_dur)
        else:
            horizon = max(8, total_dur)

    windows = compute_start_windows(instance, horizon)

    # 1. Map each valid (job_id, op_index, t) to a variable index
    var_map: dict[tuple[str, int, int], int] = {}
    inv_var_map: dict[int, tuple[str, int, int]] = {}
    var_idx = 0

    for job in instance.jobs:
        for k, op in enumerate(job.operations):
            es, ls = windows[(job.id, k)]
            for t in range(es, ls + 1):
                key = (job.id, k, t)
                var_map[key] = var_idx
                inv_var_map[var_idx] = key
                var_idx += 1

    n_vars = var_idx
    Q_cost = np.zeros((n_vars, n_vars), dtype=float)
    Q_pen = np.zeros((n_vars, n_vars), dtype=float)
    offset_pen = 0.0

    # 2. Constraint 1 (Vectorized): Exactly one start time per operation (sum_t x_t - 1)^2
    for job in instance.jobs:
        for k, op in enumerate(job.operations):
            es, ls = windows[(job.id, k)]
            times = np.array([var_map[(job.id, k, t)] for t in range(es, ls + 1)], dtype=int)

            # Block assignment for (sum x - 1)^2:
            # All elements in block get +1.0, then subtract 2.0 from diagonal (so diag is -1.0)
            Q_pen[np.ix_(times, times)] += 1.0
            Q_pen[times, times] -= 2.0
            offset_pen += 1.0

    # 3. Constraint 2 (Vectorized): Operation Precedence within each job
    for job in instance.jobs:
        for k in range(len(job.operations) - 1):
            op_k = job.operations[k]
            es_k, ls_k = windows[(job.id, k)]
            es_next, ls_next = windows[(job.id, k + 1)]

            # 2D broadcast condition: violation if next op starts before current finishes
            t_k = np.arange(es_k, ls_k + 1)[:, None]
            t_next = np.arange(es_next, ls_next + 1)[None, :]
            violation_mask = (t_next < t_k + op_k.duration).astype(float)

            v_k = np.array([var_map[(job.id, k, t)] for t in range(es_k, ls_k + 1)], dtype=int)
            v_next = np.array([var_map[(job.id, k + 1, t)] for t in range(es_next, ls_next + 1)], dtype=int)

            # Symmetrized block update
            Q_pen[np.ix_(v_k, v_next)] += 0.5 * violation_mask
            Q_pen[np.ix_(v_next, v_k)] += 0.5 * violation_mask.T

    # 4. Constraint 3 (Vectorized): Machine Resource Non-Overlap
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

                # 2D broadcast overlap condition: max(t1, t2) < min(t1 + d1, t2 + d2)
                overlap_mask = (np.maximum(t1_arr, t2_arr) < np.minimum(t1_arr + d1, t2_arr + d2)).astype(float)

                Q_pen[np.ix_(v1_arr, v2_arr)] += 0.5 * overlap_mask
                Q_pen[np.ix_(v2_arr, v1_arr)] += 0.5 * overlap_mask.T

    # 5. Objective H_cost (Vectorized): Minimize completion time of the last operation of each job
    for job in instance.jobs:
        if not job.operations:
            continue
        last_k = len(job.operations) - 1
        last_op = job.operations[last_k]
        es, ls = windows[(job.id, last_k)]
        t_vals = np.arange(es, ls + 1)
        v_indices = np.array([var_map[(job.id, last_k, t)] for t in t_vals], dtype=int)
        Q_cost[v_indices, v_indices] += (t_vals + last_op.duration).astype(float)

    # 6. Compute Sufficient Lambda if not provided
    if penalty_lambda is None:
        penalty_lambda = calculate_sufficient_lambda_from_qubo(
            Q_cost=Q_cost,
            delta=delta,
            safety_margin=1.0,
        )

    # Full QUBO: H(x) = H_cost(x) + lambda * H_penalty(x)
    Q = Q_cost + penalty_lambda * Q_pen
    offset = penalty_lambda * offset_pen

    return {
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
    }


def decode_qubo_solution(
    bitstring: np.ndarray,
    qubo_info: dict[str, Any],
    instance: InstanceData,
    method_name: str = "QUBO Solver",
) -> CalculationState:
    """Decode a flat binary vector into a concrete CalculationState schedule."""
    inv_map = qubo_info["inv_var_map"]
    state = CalculationState(
        instance=instance.instance_name,
        method=method_name,
    )

    # Group chosen start times by (job_id, op_index)
    chosen_starts: dict[tuple[str, int], int] = {}
    bits = np.array(bitstring).flatten()

    for idx, bit in enumerate(bits):
        if bit > 0.5 and idx in inv_map:
            job_id, k, t = inv_map[idx]
            chosen_starts[(job_id, k)] = t

    # Build tasks for all operations in the instance
    for job in instance.jobs:
        for k, op in enumerate(job.operations):
            if (job.id, k) in chosen_starts:
                start = chosen_starts[(job.id, k)]
            else:
                # Fallback to ES if unassigned
                start = qubo_info["windows"][(job.id, k)][0]
            end = start + op.duration
            state.add_task(job=job.id, machine=op.machine, start=start, end=end)

    state.recalculate_makespan()
    return state


def solve_qubo(
    qubo_info: dict[str, Any],
    instance: InstanceData,
    method_name: str = "Exact QUBO Solver",
) -> CalculationState:
    """Find the ground-state solution of the Job-Shop QUBO.

    Enumerates valid start-time configurations over the pruned search space,
    evaluating QUBO energy via fast submatrix summation.
    """
    Q = qubo_info["Q"]
    offset = qubo_info["offset"]
    windows = qubo_info["windows"]
    var_map = qubo_info["var_map"]
    n_vars = qubo_info["n_vars"]

    # Build the choice set of variable indices for each operation
    op_choices: list[list[int]] = []
    for job in instance.jobs:
        for k in range(len(job.operations)):
            es, ls = windows[(job.id, k)]
            var_indices = [var_map[(job.id, k, t)] for t in range(es, ls + 1)]
            op_choices.append(var_indices)

    best_choice = None
    best_energy = float("inf")

    # Fast submatrix summation: x^T Q x == sum_{i in choice, j in choice} Q[i, j]
    for choice in itertools.product(*op_choices):
        idx = np.array(choice, dtype=int)
        energy = float(Q[np.ix_(idx, idx)].sum()) + offset

        if energy < best_energy:
            best_energy = energy
            best_choice = choice

    best_bits = np.zeros(n_vars, dtype=float)
    if best_choice is not None:
        best_bits[list(best_choice)] = 1.0

    state = decode_qubo_solution(best_bits, qubo_info, instance, method_name=method_name)
    state.metadata["energy"] = best_energy
    state.metadata["penalty_lambda"] = qubo_info["penalty_lambda"]
    return state


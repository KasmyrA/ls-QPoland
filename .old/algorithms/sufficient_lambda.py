"""Calculation of sufficient penalty multiplier (lambda) for QUBO formulations.

Derived equation (as documented in LOG.md):
    lambda > (sum_i |a_i| + sum_{i<j} |b_ij|) / delta

Where:
    - a_i: linear coefficients in the objective Hamiltonian H_cost(x)
    - b_ij: quadratic interaction coefficients in H_cost(x)
    - delta: minimal penalty gap for any constraint violation:
             delta = min_{x not in F} H_penalty(x) > 0 (typically delta = 1.0)
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Union
import numpy as np

try:
    from ..entities import InstanceData
except (ImportError, ValueError):
    try:
        from controlled_code.entities import InstanceData  # type: ignore
    except ImportError:
        from entities import InstanceData  # type: ignore


def calculate_sufficient_lambda(
    a: Sequence[float],
    b: Optional[Union[Mapping[tuple[int, int], float], np.ndarray]] = None,
    delta: float = 1.0,
    safety_margin: float = 1.0,
) -> float:
    """Calculate sufficient penalty multiplier lambda from cost Hamiltonian coefficients.

    Implements:
        lambda > (sum_i |a_i| + sum_{i<j} |b_ij|) / delta

    Parameters
    ----------
    a : Sequence[float]
        Linear objective coefficients [a_0, a_1, ..., a_{n-1}].
    b : dict or ndarray, optional
        Quadratic objective coefficients b_{ij} for i < j.
        Can be a dictionary {(i, j): b_ij} or an (n, n) matrix.
    delta : float
        Minimal penalty gap delta > 0. For standard integer penalties, delta = 1.0.
    safety_margin : float
        Additional epsilon / margin added to satisfy strict inequality lambda > bound.
        Default is 1.0 (or 0.01 for fractional objectives).

    Returns
    -------
    float
        The calculated sufficient lambda.
    """
    if delta <= 0:
        raise ValueError(f"Penalty gap delta must be strictly positive, got {delta}")

    sum_linear = float(np.sum(np.abs(a)))

    sum_quadratic = 0.0
    if b is not None:
        if isinstance(b, np.ndarray):
            n = b.shape[0]
            for i in range(n):
                for j in range(i + 1, n):
                    # Combine symmetric contributions if present
                    val = b[i, j] + b[j, i]
                    sum_quadratic += abs(val)
        elif isinstance(b, Mapping):
            for (i, j), val in b.items():
                if i < j:
                    sum_quadratic += abs(val)

    bound = (sum_linear + sum_quadratic) / delta
    return float(bound + safety_margin)


def calculate_sufficient_lambda_from_qubo(
    Q_cost: np.ndarray,
    delta: float = 1.0,
    safety_margin: float = 1.0,
) -> float:
    """Calculate sufficient lambda directly from an objective QUBO matrix Q_cost.

    Parameters
    ----------
    Q_cost : np.ndarray
        The (N x N) cost Hamiltonian matrix where H_cost(x) = x^T Q_cost x.
    delta : float
        Minimal violation penalty gap (default 1.0).
    safety_margin : float
        Epsilon added for strict inequality (default 1.0).
    """
    if Q_cost.ndim != 2 or Q_cost.shape[0] != Q_cost.shape[1]:
        raise ValueError("Q_cost must be a square 2D matrix")

    n = Q_cost.shape[0]
    # Linear coefficients: diagonal entries Q[i, i] (since x_i^2 = x_i for binary)
    a = [float(Q_cost[i, i]) for i in range(n)]

    # Quadratic interaction coefficients b_{ij} for i < j
    b: dict[tuple[int, int], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            val = float(Q_cost[i, j] + Q_cost[j, i])
            if abs(val) > 1e-12:
                b[(i, j)] = val

    return calculate_sufficient_lambda(a=a, b=b, delta=delta, safety_margin=safety_margin)


def calculate_sufficient_lambda_for_instance(
    instance: InstanceData,
    horizon: Optional[int] = None,
    delta: float = 1.0,
    safety_margin: float = 1.0,
) -> float:
    """Calculate sufficient lambda for a scheduling problem InstanceData.

    For time-indexed makespan / completion time scheduling:
    Each final operation completes at time t + duration <= horizon H.
    The sum of linear cost coefficients is bounded by:
        sum_i |a_i| = sum_{final ops} sum_t (t + duration)
    With delta = 1 for discrete constraint violations.
    """
    if horizon is None:
        horizon = sum(op.duration for j in instance.jobs for op in j.operations)

    # Cost coefficients correspond to final operation completion times
    linear_coeffs: list[float] = []
    for job in instance.jobs:
        if not job.operations:
            continue
        last_op = job.operations[-1]
        for t in range(horizon + 1):
            linear_coeffs.append(float(t + last_op.duration))

    return calculate_sufficient_lambda(
        a=linear_coeffs,
        b=None,
        delta=delta,
        safety_margin=safety_margin,
    )


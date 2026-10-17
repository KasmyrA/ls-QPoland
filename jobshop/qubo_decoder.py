"""QUBO decoder for Job-Shop / ERP scheduling solutions."""
from __future__ import annotations

from typing import Any, Optional
import numpy as np

try:
    from .models import CalculationState, InstanceData
except ImportError:
    from models import CalculationState, InstanceData  # type: ignore


def decode_qubo_solution(
    bitstring: np.ndarray | list[int] | list[float],
    metadata: dict[str, Any],
    instance: Optional[InstanceData] = None,
    method_name: str = "QUBO Solver",
) -> CalculationState:
    """Decode a flat binary vector into a concrete CalculationState schedule.

    Parameters
    ----------
    bitstring : array-like
        Binary decision vector.
    metadata : dict
        Metadata dictionary from build_job_shop_qubo containing 'inv_var_map', 'windows', etc.
    instance : InstanceData, optional
        Problem instance. If not provided, retrieved from metadata['instance'].
    method_name : str
        Algorithm name for report tracking.

    Returns
    -------
    CalculationState
        Schedule state matching output_format.json and supporting dict-like indexing (e.g. decoded['makespan']).
    """
    if instance is None:
        instance = metadata.get("instance")
        if instance is None:
            raise ValueError("Instance must be provided either directly or in metadata['instance']")

    inv_map = metadata.get("inv_var_map", {})
    windows = metadata.get("windows", {})

    state = CalculationState(
        instance=instance.instance_name,
        method=method_name,
    )

    bits = np.array(bitstring).flatten()

    # Track start times chosen by the solver
    chosen_starts: dict[tuple[str, int], int] = {}

    for idx, bit in enumerate(bits):
        if bit > 0.5 and idx in inv_map:
            job_id, k, t = inv_map[idx]
            # If multiple bits are set for same op, keep first or earliest
            if (job_id, k) not in chosen_starts:
                chosen_starts[(job_id, k)] = t

    # Build tasks for all operations in the instance
    for job in instance.jobs:
        prev_end = 0
        for k, op in enumerate(job.operations):
            if (job.id, k) in chosen_starts:
                start = chosen_starts[(job.id, k)]
            else:
                # Fallback to ES or predecessor finish time
                es = windows.get((job.id, k), (0, 0))[0]
                start = max(es, prev_end)

            end = start + op.duration
            prev_end = end
            state.add_task(job=job.id, machine=op.machine, start=start, end=end)

    state.recalculate_makespan()
    is_valid, errors = state.validate(instance)
    state.metadata["status"] = "FEASIBLE" if is_valid else "INFEASIBLE"
    state.metadata["valid"] = is_valid
    state.metadata["validation_errors"] = errors

    return state


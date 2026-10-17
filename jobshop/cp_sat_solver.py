"""Exact classical Job-Shop scheduling solver using Google OR-Tools CP-SAT."""
from __future__ import annotations

from typing import Any, Optional
from ortools.sat.python import cp_model

try:
    from .models import CalculationState, InstanceData
except ImportError:
    from models import CalculationState, InstanceData  # type: ignore


def solve_cp_sat(
    instance: InstanceData,
    time_limit_seconds: float = 30.0,
) -> dict[str, Any]:
    """Solve the Job-Shop scheduling problem using OR-Tools CP-SAT.

    Parameters
    ----------
    instance : InstanceData
        Problem instance.
    time_limit_seconds : float
        Maximum solver runtime in seconds.

    Returns
    -------
    dict
        Dictionary containing 'status', 'makespan', 'tasks', 'solve_time_seconds', and 'state'.
    """
    model = cp_model.CpModel()

    # Upper bound: sum of all durations
    horizon = sum(
        op.duration
        for job in instance.jobs
        for op in job.operations
    )

    starts = {}
    ends = {}
    intervals_by_machine: dict[str, list[Any]] = {}

    # 1. Create interval variables for all operations
    for job in instance.jobs:
        for k, op in enumerate(job.operations):
            key = (job.id, k)

            start = model.new_int_var(0, horizon - op.duration, f"start_{job.id}_{k}")
            end = model.new_int_var(op.duration, horizon, f"end_{job.id}_{k}")
            interval = model.new_interval_var(start, op.duration, end, f"interval_{job.id}_{k}")

            starts[key] = start
            ends[key] = end
            intervals_by_machine.setdefault(op.machine, []).append(interval)

            # Precedence constraint within job
            if k > 0:
                previous_op = job.operations[k - 1]
                prev_key = (job.id, k - 1)
                model.add(starts[prev_key] + previous_op.duration <= start)

    # 2. Machine non-overlapping constraint
    for intervals in intervals_by_machine.values():
        model.add_no_overlap(intervals)

    # 3. Makespan objective
    makespan = model.new_int_var(0, horizon, "makespan")
    final_ends = [
        ends[(job.id, len(job.operations) - 1)]
        for job in instance.jobs
        if job.operations
    ]
    model.add_max_equality(makespan, final_ends)
    model.minimize(makespan)

    # 4. Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_seconds)
    solver.parameters.num_search_workers = 1

    status = solver.solve(model)
    status_name = solver.status_name(status)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {
            "status": status_name,
            "makespan": None,
            "tasks": [],
            "solve_time_seconds": solver.wall_time,
            "state": None,
        }

    # Extract tasks
    tasks = []
    state = CalculationState(
        instance=instance.instance_name,
        method="CP-SAT (Google OR-Tools)",
        makespan=int(solver.value(makespan)),
    )

    for job in instance.jobs:
        for k, op in enumerate(job.operations):
            key = (job.id, k)
            st = int(solver.value(starts[key]))
            et = st + op.duration
            tasks.append({
                "job_id": job.id,
                "op_index": k,
                "machine": op.machine,
                "start": st,
                "end": et,
                "duration": op.duration,
            })
            state.add_task(job=job.id, machine=op.machine, start=st, end=et)

    state.recalculate_makespan()
    is_valid, _ = state.validate(instance)
    state.metadata["status"] = "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE"

    return {
        "status": status_name,
        "makespan": solver.value(makespan),
        "tasks": sorted(tasks, key=lambda t: (t["start"], t["machine"])),
        "solve_time_seconds": solver.wall_time,
        "best_objective_bound": solver.best_objective_bound,
        "state": state,
    }


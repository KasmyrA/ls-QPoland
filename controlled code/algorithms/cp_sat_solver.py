from ortools.sat.python import cp_model


def solve_cp_sat(instance, time_limit_seconds=30):
    model = cp_model.CpModel()

    # Safe upper bound: execute all operations sequentially.
    horizon = sum(
        op.duration
        for job in instance.jobs
        for op in job.operations
    )

    starts = {}
    ends = {}
    intervals_by_machine = {}

    # Create one interval for each operation.
    for job in instance.jobs:
        for k, op in enumerate(job.operations):
            key = (job.id, k)

            start = model.new_int_var(
                0, horizon - op.duration, f"start_{job.id}_{k}"
            )
            end = model.new_int_var(
                op.duration, horizon, f"end_{job.id}_{k}"
            )

            interval = model.new_interval_var(
                start, op.duration, end, f"interval_{job.id}_{k}"
            )

            starts[key] = start
            ends[key] = end
            intervals_by_machine.setdefault(op.machine, []).append(
                interval
            )

            # Operations of a job must follow their defined order.
            if k > 0:
                previous = job.operations[k - 1]
                previous_key = (job.id, k - 1)

                model.add(
                    starts[previous_key] + previous.duration <= start
                )

    # A machine cannot process two operations simultaneously.
    for intervals in intervals_by_machine.values():
        model.add_no_overlap(intervals)

    # Standard job-shop objective: minimize makespan.
    makespan = model.new_int_var(0, horizon, "makespan")

    final_ends = [
        ends[(job.id, len(job.operations) - 1)]
        for job in instance.jobs
        if job.operations
    ]

    model.add_max_equality(makespan, final_ends)
    model.minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.num_search_workers = 1

    status = solver.solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {
            "status": solver.status_name(status),
            "tasks": [],
            "makespan": None,
        }

    tasks = []
    for job in instance.jobs:
        for k, op in enumerate(job.operations):
            key = (job.id, k)
            start = solver.value(starts[key])

            tasks.append({
                "job_id": job.id,
                "op_index": k,
                "machine": op.machine,
                "start": start,
                "end": start + op.duration,
                "duration": op.duration,
            })

    return {
        "status": solver.status_name(status),
        "makespan": solver.value(makespan),
        "tasks": sorted(
            tasks, key=lambda task: (task["start"], task["machine"])
        ),
        "solve_time_seconds": solver.wall_time,
        "best_objective_bound": solver.best_objective_bound,
    }

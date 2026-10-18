"""Breakdown re-scheduling, rolling horizon and evaluation (works with ANY solver adapter).

solver(sub_instance, fixed) -> (plan | None, info)   # make_qubo_solver(...) or make_cpsat_solver(...)
Plan = {(job, op_index): (unit, start, end)}
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from jsons.entities import CalculationState, InstanceData, Job
from .schedule_utils import Fixed, Plan, op_options, overlap, plan_cost, validate_plan
from .heuristics import greedy_schedule

Solver = Callable[[InstanceData, Fixed], tuple[Optional[Plan], dict]]


# ------------------------------------------------------------------ state <-> plan
def plan_from_state(state: CalculationState, instance: InstanceData) -> Plan:
    """Previous output JSON -> Plan. Tasks without 'unit' are assigned to a free unit of their type."""
    units_of: dict[str, list[str]] = {}
    for u, t in instance.unit_type.items():
        units_of.setdefault(t, []).append(u)
    for job in instance.jobs:
        n_tasks = len(state.get_job_schedule(job.id))
        if n_tasks != len(job.operations):
            raise ValueError(f"{job.id}: {n_tasks} tasks in previous schedule, {len(job.operations)} operations")
    busy: dict[str, list[tuple[int, int]]] = {}
    assigned: dict[tuple[str, int, int], str] = {}
    for task in sorted(state.schedule, key=lambda t: (t.start, t.job)):
        unit = task.unit
        if not unit:
            cands = units_of.get(task.machine, [task.machine])
            unit = next((u for u in cands
                         if all(not overlap(task.start, task.end, a, b) for a, b in busy.get(u, []))), cands[0])
        busy.setdefault(unit, []).append((task.start, task.end))
        assigned[(task.job, task.start, task.end)] = unit
    plan: Plan = {}
    for job in instance.jobs:
        for k, task in enumerate(state.get_job_schedule(job.id)):
            plan[(job.id, k)] = (assigned[(job.id, task.start, task.end)], task.start, task.end)
    return plan


def state_from_plan(plan: Plan, instance: InstanceData, method: str, metadata: Optional[dict] = None) -> CalculationState:
    state = CalculationState(instance=instance.instance_name, method=method)
    for (jid, k), (u, s, e) in sorted(plan.items()):
        typ = instance.unit_type.get(u, u)
        state.add_task(job=jid, machine=typ, start=s, end=e, unit=u if instance.unit_type else "")
    state.recalculate_makespan()
    state.metadata.update(metadata or {})
    return state


# ------------------------------------------------------------------ sub-problem
def _build_sub(instance, committed: Plan, free: dict[str, tuple[int, int]], now: int,
               outages: dict, partial: dict):
    sub_jobs, release, due_end, offsets = [], {}, {}, {}
    for jid, (a, b) in free.items():
        job = instance.get_job(jid)
        if b <= a:
            continue
        sub_jobs.append(Job(id=jid, operations=job.operations[a:b]))
        offsets[jid] = a
        r = now
        if a > 0:
            r = max(r, committed[(jid, a - 1)][2])
        release[(jid, 0)] = r
        if (jid, b) in committed:
            due_end[(jid, b - a - 1)] = committed[(jid, b)][1]
    max_L = max([x + y for x, y in instance.limits.values()] + [0])
    frozen: dict[str, list] = {}
    for (jid, k), (u, s, e) in committed.items():
        if e > now - max_L:
            frozen.setdefault(u, []).append((s, e))
    for u, ivs in partial.items():
        frozen.setdefault(u, []).extend(ivs)
    fixed = Fixed(release=release, due_end=due_end, outages=outages, committed=frozen)
    sub = InstanceData(instance_name=instance.instance_name, problem=instance.problem, jobs=sub_jobs,
                       machines=instance.machines, unit_type=instance.unit_type, limits=instance.limits)
    return sub, fixed, offsets


def solve_partial(instance, committed: Plan, free, now, outages, partial, solver: Solver):
    sub, fixed, offsets = _build_sub(instance, committed, free, now, outages, partial)
    if not sub.jobs:
        return {}, {"note": "nothing to solve"}
    plan, info = solver(sub, fixed)
    if plan is None:
        return None, info
    return {(jid, offsets[jid] + k): v for (jid, k), v in plan.items()}, info


# ------------------------------------------------------------------ breakdown re-scheduling
def _classify(instance, prev: Plan, now: int, outages: dict, window: Optional[int]):
    committed: Plan = {}
    partial: dict[str, list] = {}
    free: dict[str, tuple[int, int]] = {}
    disrupted = 0
    for job in instance.jobs:
        n = len(job.operations)
        a = n
        for k in range(n):
            u, s, e = prev[(job.id, k)]
            hit = [bs for bs, be in outages.get(u, []) if overlap(s, e, bs, be)]
            if e <= now or (s < now and not hit):
                continue
            a = k
            if s < now and hit:                                  # interrupted: work done so far still counts
                w_end = min(e, min(hit))
                if w_end > s:
                    partial.setdefault(u, []).append((s, w_end))
            break
        last_dis = -1
        for k in range(a, n):
            u, s, e = prev[(job.id, k)]
            if any(overlap(s, e, bs, be) for bs, be in outages.get(u, [])):
                last_dis = k
                disrupted += 1
        b = n
        if window is not None:
            b0 = next((k for k in range(a, n) if prev[(job.id, k)][1] >= now + window), n)
            b = max(b0, last_dis + 1)
        for k in list(range(0, a)) + list(range(b, n)):
            committed[(job.id, k)] = prev[(job.id, k)]
        free[job.id] = (a, b)
    return committed, partial, free, disrupted


def reschedule(instance: InstanceData, prev: Plan, now: int, outages: dict, solver: Solver,
               window: Optional[int] = None):
    """Re-plan after breakdowns. Done / in-progress-on-healthy-unit operations are frozen,
    interrupted ones restart. `window` (time units) freezes operations starting later than now+window.
    Returns (new_plan, info)."""
    info: dict[str, Any] = {}
    for w in ([window, None] if window is not None else [None]):
        committed, partial, free, disrupted = _classify(instance, prev, now, outages, w)
        res, sinfo = solve_partial(instance, committed, free, now, outages, partial, solver)
        info = {"window_requested": window, "window_used": w, "ops_disrupted": disrupted, **sinfo}
        if res is not None:
            new = dict(committed); new.update(res)
            errs = validate_plan(instance, new, Fixed(outages=outages), check_duration=False)
            info["violations"] = errs
            if not errs:
                return new, info
    info["fallback"] = "right_shift"
    return right_shift(instance, prev, now, outages), info


def right_shift(instance, prev: Plan, now: int, outages: dict) -> Plan:
    """Baseline: keep units and order, only push operations later until they fit."""
    committed, partial, free, _ = _classify(instance, prev, now, outages, None)
    sub, fixed, offsets = _build_sub(instance, committed, free, now, outages, partial)
    follow = {(jid, k): (prev[(jid, offsets[jid] + k)][0], prev[(jid, offsets[jid] + k)][1])
              for jid in offsets for k in range(len(instance.get_job(jid).operations) - offsets[jid])}
    plan = greedy_schedule(sub, fixed, now=now, follow=follow) if sub.jobs else {}
    new = dict(committed)
    for (jid, k), v in (plan or {}).items():
        new[(jid, offsets[jid] + k)] = v
    return new


# ------------------------------------------------------------------ rolling horizon
def rolling_schedule(instance: InstanceData, solver: Solver, step: int = 3, lookahead: int = 2,
                     outages: Optional[dict] = None, max_iter: int = 200):
    """Plan from scratch in slices: each round solves only the next `lookahead` operations of every
    job (small QUBO), commits what starts before now+step, then moves `now` forward."""
    outages = outages or {}
    committed: Plan = {}
    progress = {j.id: 0 for j in instance.jobs}
    n_ops = {j.id: len(j.operations) for j in instance.jobs}
    now, rounds, sizes = 0, 0, []
    while any(progress[j] < n_ops[j] for j in progress):
        rounds += 1
        if rounds > max_iter:
            raise RuntimeError("rolling horizon did not converge")
        free = {j: (progress[j], min(n_ops[j], progress[j] + lookahead)) for j in progress if progress[j] < n_ops[j]}
        res, info = solve_partial(instance, committed, free, now, outages, {}, solver)
        if res is None:
            raise RuntimeError(f"sub-problem at now={now} has no feasible solution: {info}")
        sizes.append(info.get("n_vars"))
        if all(b == n_ops[j] for j, (a, b) in free.items()):     # lookahead covers everything left
            committed.update(res)
            break
        now += step
        for j in free:
            while progress[j] < n_ops[j] and (j, progress[j]) in res and res[(j, progress[j])][1] < now:
                committed[(j, progress[j])] = res[(j, progress[j])]
                progress[j] += 1
    errs = validate_plan(instance, committed, Fixed(outages=outages))
    return committed, {"rounds": rounds, "n_vars_per_round": sizes, "violations": errs}


# ------------------------------------------------------------------ evaluation
def evaluate(instance, prev: Plan, new: Plan, baseline: Plan, outages: dict) -> dict[str, Any]:
    def mk(p): return max(e for _, _, e in p.values())
    def tot(p): return sum(p[(j.id, len(j.operations) - 1)][2] for j in instance.jobs)
    disrupted = sum(1 for _, (u, s, e) in prev.items() if any(overlap(s, e, bs, be) for bs, be in outages.get(u, [])))
    return {
        "makespan_before": mk(prev),
        "makespan_right_shift": mk(baseline),
        "makespan_rescheduled": mk(new),
        "sum_completion_before": tot(prev),
        "sum_completion_right_shift": tot(baseline),
        "sum_completion_rescheduled": tot(new),
        "ops_disrupted_by_outage": disrupted,
        "ops_moved": sum(1 for k in new if new[k][:2] != prev[k][:2]),
        "violations": validate_plan(instance, new, Fixed(outages=outages), check_duration=False),
    }
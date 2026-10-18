"""Shared helpers: fixed constraints, plan validation, plan cost.

Plan = {(job_id, op_index): (unit, start, end)}

Duty-cycle semantics (same in QUBO, CP-SAT and the validator):
  a machine unit with limits (X, Y) may work at most X time units inside ANY window of
  X+Y consecutive time units, i.e. after X units of continuous work it needs Y units of rest.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from jsons.entities import InstanceData

Interval = tuple[int, int]
Plan = dict[tuple[str, int], tuple[str, int, int]]


@dataclass
class Fixed:
    """Constraints coming from the outside world (breakdowns, frozen operations, ...)."""
    release: dict[tuple[str, int], int] = field(default_factory=dict)   # earliest START of op
    due_end: dict[tuple[str, int], int] = field(default_factory=dict)   # latest END of op
    outages: dict[str, list[Interval]] = field(default_factory=dict)    # unit unavailable
    committed: dict[str, list[Interval]] = field(default_factory=dict)  # frozen WORK on unit

    def blocked(self, unit: str) -> list[Interval]:
        return list(self.outages.get(unit, [])) + list(self.committed.get(unit, []))


def op_options(op) -> dict[str, int]:
    """{unit: duration} for an operation (falls back to its single machine)."""
    m = getattr(op, "machines", None)
    if m:
        return {str(u): int(d) for u, d in m.items()}
    return {op.machine: int(op.duration)}


def overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def plan_cost(plan: Plan, instance: InstanceData, objective: str = "makespan") -> int:
    ends = [plan[(j.id, len(j.operations) - 1)][2] for j in instance.jobs if j.operations]
    if not ends:
        return 0
    return max(ends) if objective == "makespan" else sum(ends)


def validate_plan(instance: InstanceData, plan: Plan, fixed: Optional[Fixed] = None,
                  check_duration: bool = True) -> list[str]:
    """Return a list of violations (empty list = feasible)."""
    fixed = fixed or Fixed()
    errs: list[str] = []

    for job in instance.jobs:
        for k, op in enumerate(job.operations):
            key = (job.id, k)
            if key not in plan:
                errs.append(f"{job.id}/{k}: missing in plan")
                continue
            u, s, e = plan[key]
            opts = op_options(op)
            if u not in opts:
                errs.append(f"{job.id}/{k}: unit {u} not allowed")
            elif check_duration and e - s != opts[u]:
                errs.append(f"{job.id}/{k}: duration {e - s} != {opts[u]} on {u}")
            if s < fixed.release.get(key, -10**9):
                errs.append(f"{job.id}/{k}: starts before release")
            if e > fixed.due_end.get(key, 10**9):
                errs.append(f"{job.id}/{k}: ends after due_end")
        for k in range(len(job.operations) - 1):
            a, b = (job.id, k), (job.id, k + 1)
            if a in plan and b in plan and plan[a][2] > plan[b][1]:
                errs.append(f"{job.id}: precedence {k}->{k + 1}")

    per_unit: dict[str, list[tuple[int, int, str]]] = {}
    for (jid, k), (u, s, e) in plan.items():
        per_unit.setdefault(u, []).append((s, e, f"{jid}/{k}"))

    for u, ivs in per_unit.items():
        ivs.sort()
        for i in range(len(ivs) - 1):
            if ivs[i][1] > ivs[i + 1][0]:
                errs.append(f"{u}: overlap {ivs[i][2]} / {ivs[i + 1][2]}")
        for s, e, lab in ivs:
            for bs, be in fixed.outages.get(u, []):
                if overlap(s, e, bs, be):
                    errs.append(f"{u}: {lab} overlaps outage [{bs},{be})")
            for bs, be in fixed.committed.get(u, []):
                if overlap(s, e, bs, be):
                    errs.append(f"{u}: {lab} overlaps frozen work [{bs},{be})")

    for u, (X, Y) in instance.limits.items():
        work = [(s, e) for s, e, _ in per_unit.get(u, [])] + list(fixed.committed.get(u, []))
        if not work:
            continue
        H = max(e for _, e in work)
        busy = np.zeros(H + 1, dtype=int)
        for s, e in work:
            busy[s:e] += 1
        cs = np.concatenate([[0], np.cumsum(busy)])
        L = X + Y
        for s in range(max(1, H - L + 1)):
            if cs[min(s + L, H)] - cs[s] > X:
                errs.append(f"{u}: duty cycle (max {X} work per {L}) violated in window starting {s}")
                break
    return errs
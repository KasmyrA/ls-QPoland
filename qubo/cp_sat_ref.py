"""Exact classical reference (OR-Tools CP-SAT) with the SAME semantics as the QUBO.

Time-indexed model (one Boolean per unit/start) so that every feature - alternative units,
unit-dependent durations, duty cycles, breakdowns, frozen operations - means exactly the same
thing as in qubo.py. Use it to verify the QUBO and as the 'exact' backend for small problems.
"""
from __future__ import annotations

from typing import Optional

from ortools.sat.python import cp_model

try:
    from ..jsons.entities import InstanceData
    from .schedule_utils import Fixed, Plan, op_options, overlap
    from .heuristics import greedy_horizon
except ImportError:
    from .entities import InstanceData  # type: ignore
    from schedule_utils import Fixed, Plan, op_options, overlap  # type: ignore
    from heuristics import greedy_horizon  # type: ignore


def solve_cp_sat_ref(instance: InstanceData, fixed: Optional[Fixed] = None, objective: str = "makespan",
                     time_limit: float = 30.0, horizon: Optional[int] = None):
    """Returns (plan | None, status_name, objective_value | None)."""
    fixed = fixed or Fixed()
    now = min(fixed.release.values(), default=0)
    H = horizon or greedy_horizon(instance, fixed, now=now) + 2
    m = cp_model.CpModel()

    X: dict[tuple[str, int], list] = {}
    for job in instance.jobs:
        for k, op in enumerate(job.operations):
            lst = []
            for u, d in op_options(op).items():
                blocked = fixed.blocked(u)
                t_min = fixed.release.get((job.id, k), now)
                t_max = min(H, fixed.due_end.get((job.id, k), H)) - d
                for t in range(t_min, t_max + 1):
                    if any(overlap(t, t + d, bs, be) for bs, be in blocked):
                        continue
                    lst.append((m.new_bool_var(f"x_{job.id}_{k}_{u}_{t}"), u, t, d))
            if not lst:
                return None, "INFEASIBLE", None
            X[(job.id, k)] = lst
            m.add_exactly_one([v for v, *_ in lst])

    def expr(key, what):
        return sum((t + d if what == "end" else t) * v for v, _, t, d in X[key])

    for job in instance.jobs:
        for k in range(len(job.operations) - 1):
            m.add(expr((job.id, k), "end") <= expr((job.id, k + 1), "start"))

    by_unit: dict[str, list] = {}
    for lst in X.values():
        for v, u, t, d in lst:
            by_unit.setdefault(u, []).append((v, t, d))
    for u, lst in by_unit.items():
        lo = min(t for _, t, _ in lst)
        for slot in range(lo, H):
            terms = [v for v, t, d in lst if t <= slot < t + d]
            if len(terms) > 1:
                m.add_at_most_one(terms)

    for u, (Xw, Yr) in instance.limits.items():
        lst = by_unit.get(u)
        if not lst:
            continue
        L = Xw + Yr
        frozen = fixed.committed.get(u, [])
        H_d = max([H] + [e for _, e in frozen])
        for s in range(0, max(1, H_d - L + 1)):
            e = min(s + L, H_d)
            terms = [overlap(t, t + d, s, e) * v for v, t, d in lst if overlap(t, t + d, s, e) > 0]
            if terms:
                m.add(sum(terms) <= Xw - sum(overlap(a, b, s, e) for a, b in frozen))

    ends = [expr((j.id, len(j.operations) - 1), "end") for j in instance.jobs if j.operations]
    if objective == "makespan":
        M = m.new_int_var(0, H + 10, "makespan")
        for en in ends:
            m.add(M >= en)
        m.minimize(M)
    else:
        m.minimize(sum(ends))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = 8
    status = solver.solve(m)
    name = solver.status_name(status)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, name, None
    plan: Plan = {}
    for key, lst in X.items():
        for v, u, t, d in lst:
            if solver.value(v):
                plan[key] = (u, t, t + d)
    return plan, name, solver.objective_value


def make_cpsat_solver(objective: str = "makespan", time_limit: float = 30.0):
    """Same adapter signature as qubo.make_qubo_solver."""
    def solve(instance, fixed=None):
        plan, status, obj = solve_cp_sat_ref(instance, fixed, objective, time_limit)
        return plan, {"status": status, "objective": obj}
    return solve
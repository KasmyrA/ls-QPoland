"""High-level entry points.

  solve_basic     instance.json                                     -> schedule.json
  solve_extended  instance.json + machines.json [+ events.json [+ previous schedule.json]]
                                                                    -> schedule.json (+ report.json)
  solve_rolling   instance.json [+ machines.json]                   -> schedule.json (slice by slice)

backend: "sa"      local simulated annealing on the QUBO (offline stand-in for the cloud solver)
         "exact"   brute-force QUBO ground state (<= ~24 variables)
         "sbm" / "veloxq"   VeloxQ cloud (needs VELOX_TOKEN)
         "cpsat"   exact classical reference (OR-Tools)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

try:
    from ..jsons.load import load_instance, load_schedule
    from ..jsons.save import save_schedule
    from ..jsons.machines import expand_instance, load_events, load_machine_config, resolve_outages
    from .schedule_utils import Fixed, plan_cost, validate_plan
    from .qubo import make_qubo_solver
    from .samplers import make_exact_sampler, make_sa_sampler
    from .cp_sat_ref import make_cpsat_solver
    from .rolling import (evaluate, plan_from_state, reschedule, right_shift,
                          rolling_schedule, state_from_plan)
except ImportError:
    from loader import load_instance, load_schedule  # type: ignore
    from saver import save_schedule  # type: ignore
    from machines import expand_instance, load_events, load_machine_config, resolve_outages  # type: ignore
    from schedule_utils import Fixed, plan_cost, validate_plan  # type: ignore
    from qubo import make_qubo_solver  # type: ignore
    from samplers import make_exact_sampler, make_sa_sampler  # type: ignore
    from cp_sat_ref import make_cpsat_solver  # type: ignore
    from rolling import (evaluate, plan_from_state, reschedule, right_shift,  # type: ignore
                         rolling_schedule, state_from_plan)


def build_solver(backend: str = "sa", objective: str = "makespan", *, sa_reads: int = 96, sa_sweeps: int = 600,
                 seed: int = 0, velox_reps: int = 2000, velox_runs: int = 3, velox_timeout: int = 60,
                 cpsat_time_limit: float = 30.0):
    b = backend.lower()
    if b == "cpsat":
        return make_cpsat_solver(objective, cpsat_time_limit), "CP-SAT (OR-Tools, time-indexed)"
    if b == "exact":
        sampler, name = make_exact_sampler(), "QUBO Exact Ground-State"
    elif b == "sa":
        sampler, name = make_sa_sampler(sa_reads, sa_sweeps, seed), "QUBO + simulated annealing (local)"
    elif b in ("sbm", "veloxq"):
        try:
            from .veloxq_solver import make_veloxq_sampler
        except ImportError:
            from veloxq_solver import make_veloxq_sampler  # type: ignore
        sampler = make_veloxq_sampler(b, velox_reps, velox_timeout, runs=velox_runs)
        name = "VeloxQ SBM (Simulated Bifurcation)" if b == "sbm" else "VeloxQ H100 Solver"
    else:
        raise ValueError(f"unknown backend: {backend}")
    return make_qubo_solver(sampler, objective=objective), name


def _finish(plan, info, inst, method, out_path, extra=None):
    if plan is None:
        raise RuntimeError(f"no feasible schedule found ({info}). Increase reads/num_rep/runs or the horizon.")
    errs = validate_plan(inst, plan)
    state = state_from_plan(plan, inst, method, {"info": info, "violations": errs, **(extra or {})})
    if out_path:
        save_schedule(state, out_path)
    return state


# ---------------------------------------------------------------- path 1: instance JSON only
def solve_basic(instance_path, backend="sa", objective="makespan", out_path=None, **kw):
    inst = load_instance(instance_path)
    solver, method = build_solver(backend, objective, **kw)
    plan, info = solver(inst, Fixed())
    return _finish(plan, info, inst, method, out_path)


# ---------------------------------------------------------------- path 2: + machines (+ events)
def solve_extended(instance_path, machines_path, events_path=None, prev_path=None, backend="sa",
                   objective="makespan", window: Optional[int] = None, out_path=None, report_path=None, **kw):
    inst = expand_instance(load_instance(instance_path), load_machine_config(machines_path))
    solver, method = build_solver(backend, objective, **kw)

    if events_path is None:
        plan, info = solver(inst, Fixed())
        return _finish(plan, info, inst, method, out_path), None

    now, outages = resolve_outages(inst, load_events(events_path))
    if prev_path:
        prev = plan_from_state(load_schedule(prev_path), inst)
    else:                                              # no previous output: plan first, ignoring the breakdown
        prev, _ = solver(inst, Fixed())
        if prev is None:
            raise RuntimeError("could not build the initial schedule")
    new, info = reschedule(inst, prev, now, outages, solver, window=window)
    report = evaluate(inst, prev, new, right_shift(inst, prev, now, outages), outages)
    report.update(now=now, outages={u: v for u, v in outages.items()}, solver_info=info)
    state = state_from_plan(new, inst, f"{method} + reschedule", {"info": info, "report": report, "violations": report["violations"]})
    if out_path:
        save_schedule(state, out_path)
    if report_path:
        save_schedule({k: (v if k != "outages" else {u: [list(i) for i in iv] for u, iv in v.items()})
                       for k, v in report.items() if k != "solver_info"} | {"solver_info": _plain(info)}, report_path)
    return state, report


# ---------------------------------------------------------------- rolling horizon
def solve_rolling(instance_path, machines_path=None, events_path=None, backend="sa", objective="sum_completion",
                  step=3, lookahead=2, out_path=None, **kw):
    inst = load_instance(instance_path)
    outages = {}
    if machines_path:
        inst = expand_instance(inst, load_machine_config(machines_path))
        if events_path:
            _, outages = resolve_outages(inst, load_events(events_path))
    solver, method = build_solver(backend, objective, **kw)
    plan, info = rolling_schedule(inst, solver, step=step, lookahead=lookahead, outages=outages)
    return _finish(plan, info, inst, f"{method} (rolling horizon)", out_path)


def _plain(x):
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if hasattr(x, "item"):
        return x.item()
    return x
"""Self-contained test runner for the ERP / job-shop framework.

Everything runs in-process through the framework modules (no other scripts are called).

Groups
  base3x3    3x3 instance with machines: CP-SAT, SA, rolling (CP-SAT, SA)
  breakdown  3x3 with a breakdown: reschedule with CP-SAT and SA, plus right-shift baseline
  quantum    2x2 cases with QAOA (gate-based, statevector, <= 16 qubits), CP-SAT as reference
  real       5x5, 5x5 breakdown, 6x3 duty cycles, 8x4 breakdown: CP-SAT, SA, rolling, reschedule
  rolling    12x5 (and 15x6 with --with-big): rolling with CP-SAT, SA, optional VeloxQ

VeloxQ (cloud) runs only with --veloxq (needs veloxq_sdk and VELOX_TOKEN).
Every run writes results/<group>/<case>__<mode>__<solver>.json (schedule in output_format.json layout)
and .meta.json (record). results/summary.csv + results/summary.json collect all records.

Output format
  --format plain      (default) exactly the keys of output_format.json:
                      {"instance", "method", "makespan", "schedule": [{"job","machine","start","end"}]}
  --format framework  same, plus "unit" per task (T1, T2, ...) when the instance has physical units.

Usage
  python run_all_tests.py                       # all groups, local solvers
  python run_all_tests.py --only quantum real   # selected groups
  python run_all_tests.py --veloxq              # add VeloxQ runs
  python run_all_tests.py --with-big            # add 15x6 rolling (slow)
  python run_all_tests.py --format framework    # keep unit field (T1, T2) in the output
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from jsons.load import load_instance, load_schedule
from jsons.machines import expand_instance, load_events, load_machine_config, resolve_outages
from qubo.qubo import make_qubo_solver
from qubo.rolling import plan_from_state, reschedule, right_shift, rolling_schedule
from qubo.samplers import make_sa_sampler
from qubo.schedule_utils import Fixed, plan_cost, validate_plan

ROOT = Path("results")
CP_TIME = 60.0
SA_READS, SA_SWEEPS = 48, 300
QAOA_REPS, QAOA_MAXITER, QAOA_TOPK, QAOA_MAX_QUBITS = 2, 150, 256, 16
FORMATS = ("plain", "framework")
OUTPUT_FORMAT = "plain"
METHOD_NAMES = {
    "cpsat": "CP-SAT (OR-Tools)",
    "sa": "QUBO simulated annealing (local)",
    "qaoa": "QAOA (state-vector simulator)",
    "veloxq": "VeloxQ SBM (cloud)",
}


@dataclass
class Case:
    name: str
    group: str
    instance: str
    machines: Optional[str] = None
    events: Optional[str] = None
    objective: str = "makespan"
    modes: list = field(default_factory=lambda: ["plan"])          # plan | reschedule | rolling
    solvers: list = field(default_factory=lambda: ["cpsat", "sa"])


CASES = [
    # --- group base3x3
    Case("3x3_base", "base3x3", "tests/3x3/3x3_base_instance.json", "tests/3x3/3x3_base_machines.json",
         solvers=["cpsat", "sa"], modes=["plan", "rolling"]),
    # --- group breakdown (fixtures are stored under hybrid)
    Case("3x3_breakdown", "breakdown", "tests/hybrid/3x3_base_instance.json",
         "tests/hybrid/3x3_breakdown_machines.json", "tests/hybrid/3x3_breakdown_events.json",
         solvers=["cpsat", "sa"], modes=["reschedule"]),
    # --- group quantum (QAOA on 2x2)
    Case("q2x2_basic", "quantum", "tests/quantum_2x2/q2x2_basic_instance.json",
         solvers=["qaoa", "cpsat"], modes=["plan"]),
    # --- group real
    Case("5x5", "real", "tests/stress/stress_5x5_instance.json", "tests/stress/stress_5x5_machines.json",
         solvers=["cpsat", "sa"], modes=["plan", "rolling"]),
    Case("5x5_breakdown", "real", "tests/stress/stress_5x5_instance.json", "tests/stress/stress_5x5_machines.json",
         "tests/real/real_5x5_breakdown_events.json", solvers=["cpsat", "sa"], modes=["reschedule"]),
    Case("6x3_duty", "real", "tests/6x3_duty_instance.json", "tests/6x3_duty_machines.json",
         solvers=["cpsat", "sa"], modes=["plan", "rolling"]),
    Case("8x4_breakdown", "real", "tests/8x4_breakdown_instance.json", "tests/8x4_breakdown_machines.json",
         "tests/8x4_breakdown_events.json", solvers=["cpsat", "sa"], modes=["reschedule"]),
    # --- group rolling
    Case("roll_12x5", "rolling", "tests/rolling/roll_12x5_instance.json", "tests/rolling/roll_12x5_machines.json",
         objective="sum_completion", solvers=["cpsat", "sa"], modes=["rolling"]),
]
BIG = Case("roll_15x6", "rolling", "tests/rolling/roll_15x6_instance.json", "tests/rolling/roll_15x6_machines.json",
           objective="sum_completion", solvers=["cpsat"], modes=["rolling"])
GROUPS = ["base3x3", "breakdown", "quantum", "real", "rolling"]
VELOX_SELECTED: set = set()


# ------------------------------------------------------------------ output format
def build_output(plan, inst, instance_name: str, method: str, fmt: str) -> dict:
    """plan -> dict in the output_format.json layout."""
    tasks = []
    for (jid, k), (unit, s, e) in plan.items():
        machine = inst.unit_type.get(unit, unit) if inst.unit_type else unit
        task = {"job": jid, "machine": machine, "start": int(s), "end": int(e)}
        if fmt == "framework" and inst.unit_type:
            task["unit"] = unit
        tasks.append(task)
    tasks.sort(key=lambda t: (t["start"], t["job"], t["machine"]))
    makespan = max((t["end"] for t in tasks), default=0)
    return {"instance": instance_name, "method": method, "makespan": int(makespan), "schedule": tasks}


def write_output(path: Path, plan, inst, instance_name: str, method: str, fmt: str) -> dict:
    """Writes the output file, then reloads it and checks that it reads back as a feasible plan."""
    out = build_output(plan, inst, instance_name, method, fmt)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=4, ensure_ascii=False), encoding="utf-8")
    back = plan_from_state(load_schedule(str(path)), inst)
    errs = validate_plan(inst, back, Fixed())
    return {"reload_ok": not errs, "reload_violations": len(errs)}


# ------------------------------------------------------------------ solvers
def make_solver(name: str, objective: str):
    if name == "cpsat":
        from qubo.cp_sat_ref import make_cpsat_solver
        return make_cpsat_solver(objective, CP_TIME)
    if name == "sa":
        return make_qubo_solver(make_sa_sampler(SA_READS, SA_SWEEPS, seed=0), objective=objective)
    if name == "qaoa":
        from qubo.qaoa_solver import make_qaoa_sampler
        return make_qubo_solver(make_qaoa_sampler(QAOA_REPS, QAOA_MAXITER, top_k=QAOA_TOPK,
                                                  max_qubits=QAOA_MAX_QUBITS), objective=objective)
    if name == "veloxq":
        from qubo.veloxq_solver import make_veloxq_sampler
        return make_qubo_solver(make_veloxq_sampler("veloxq", 2000, 60, runs=3),
                            objective=objective, lambda_mode="sufficient")
    raise ValueError(name)


def load_case(case: Case):
    inst = load_instance(case.instance)
    if case.machines:
        inst = expand_instance(inst, load_machine_config(case.machines))
    now, outages = None, {}
    if case.events:
        now, outages = resolve_outages(inst, load_events(case.events))
    return inst, now, outages


# ------------------------------------------------------------------ one run
def run_one(case: Case, mode: str, solver_name: str, inst, now, outages, base, ref_cost) -> dict:
    rec = dict(group=case.group, case=case.name, mode=mode, solver=solver_name,
               objective=case.objective, reference_cost=ref_cost)
    t0 = time.time()
    folder = ROOT / case.group
    folder.mkdir(parents=True, exist_ok=True)
    try:
        solver_obj = make_solver(solver_name, case.objective)
        if mode == "plan":
            plan, info = solver_obj(inst, Fixed())
        elif mode == "reschedule":
            plan, info = reschedule(inst, base, now, outages, solver_obj)
            rec["right_shift_cost"] = plan_cost(right_shift(inst, base, now, outages), inst, case.objective)
        else:  # rolling
            plan, info = rolling_schedule(inst, solver_obj, step=3, lookahead=2, outages=outages)
        rec["n_vars"] = info.get("n_vars")
        if plan is None:
            rec["status"] = "no_solution"
        else:
            errs = validate_plan(inst, plan, Fixed(outages=outages), check_duration=True)
            cost = plan_cost(plan, inst, case.objective)
            rec.update(status="ok" if not errs else "infeasible", cost=cost, violations=len(errs),
                       gap_pct=round(100 * (cost - ref_cost) / ref_cost, 1) if ref_cost else None)
            method = f"{METHOD_NAMES.get(solver_name, solver_name)}, {mode}"
            rec.update(write_output(folder / f"{case.name}__{mode}__{solver_name}.json",
                                    plan, inst, case.name, method, OUTPUT_FORMAT))
    except Exception as exc:
        rec.update(status="error", error=f"{type(exc).__name__}: {str(exc)[:240]}")
    rec["time_s"] = round(time.time() - t0, 1)
    (folder / f"{case.name}__{mode}__{solver_name}.meta.json").write_text(
        json.dumps(rec, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return rec


def run_case(case: Case, rows: list) -> None:
    from qubo.cp_sat_ref import make_cpsat_solver, solve_cp_sat_ref

    inst, now, outages = load_case(case)
    plan0, st0, _ = solve_cp_sat_ref(inst, objective=case.objective, time_limit=CP_TIME)
    if plan0 is None:
        rows.append(dict(group=case.group, case=case.name, mode="reference", solver="cpsat", status=st0))
        return
    base, ref_plan = plan0, plan0
    if outages:
        ref_plan, _ = reschedule(inst, plan0, now, outages, make_cpsat_solver(case.objective, CP_TIME))
    ref_cost = plan_cost(ref_plan, inst, case.objective)
    print(f"[{case.group}/{case.name}] reference CP-SAT ({case.objective}): {ref_cost} ({st0})", flush=True)

    for mode in case.modes:
        if (mode == "reschedule") != bool(outages):
            continue                                  # reschedule only with events, plan/rolling only without
        solvers = list(case.solvers) + (["veloxq"] if "veloxq" in VELOX_SELECTED else [])
        for s in solvers:
            rec = run_one(case, mode, s, inst, now, outages, base, ref_cost)
            rows.append(rec)
            print(f"  {mode:>10} {s:<7} {rec.get('status'):<12} cost={rec.get('cost')} "
                  f"gap={rec.get('gap_pct')}% t={rec['time_s']}s {rec.get('error', '')}", flush=True)


def write_summary(rows: list) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    keys = ["group", "case", "mode", "solver", "status", "cost", "reference_cost", "gap_pct",
            "violations", "right_shift_cost", "n_vars", "time_s", "error"]
    with open(ROOT / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    (ROOT / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False, default=str),
                                       encoding="utf-8")
    print(f"summary: {ROOT / 'summary.csv'} ({len(rows)} rows)")


def main():
    global CP_TIME, OUTPUT_FORMAT
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*", choices=GROUPS, default=None)
    p.add_argument("--veloxq", action="store_true")
    p.add_argument("--with-big", action="store_true", help="add 15x6 rolling (slow)")
    p.add_argument("--cp-time", type=float, default=60.0)
    p.add_argument("--format", choices=FORMATS, default="plain",
                   help="plain = output_format.json keys only; framework = adds unit per task")
    a = p.parse_args()
    CP_TIME = a.cp_time
    OUTPUT_FORMAT = a.format
    if a.veloxq:
        VELOX_SELECTED.add("veloxq")

    cases = CASES + ([BIG] if a.with_big else [])
    wanted = a.only or GROUPS
    rows: list = []
    for case in cases:
        if case.group not in wanted:
            continue
        try:
            run_case(case, rows)
        except Exception as exc:
            rows.append(dict(group=case.group, case=case.name, status="error",
                             error=f"{type(exc).__name__}: {str(exc)[:240]}"))
            traceback.print_exc()
        write_summary(rows)          # keep partial results if the run is interrupted
    write_summary(rows)


if __name__ == "__main__":
    main()
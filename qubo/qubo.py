"""QUBO builder for Job-Shop / ERP scheduling (one engine for every scenario).

Binary variables
  x[(job, k, unit, t)] = 1  <=> operation k of `job` starts on `unit` at time t
  y[T]                = 1   <=> makespan equals T            (objective="makespan" only)
  z[...]                    slack bits of the duty-cycle inequalities (not decision variables)

Penalty terms (all quadratic)
  1. exactly one (unit, start) per operation
  2. precedence inside a job (independent of the chosen units)
  3. no overlap on the same unit (only variables of the same unit interact)
  4. duty cycle: sum of work in every window of X+Y time units <= X   (slack encoded)
  5. makespan: y one-hot and  x_last(job, completion c) * y_T  forbidden when c > T

Outside constraints (breakdowns, frozen operations, rolling horizon) come through `Fixed`:
  release / due_end shrink start windows, outages and committed intervals remove variables,
  committed work is subtracted from the duty-cycle budget of each window.
"""
from __future__ import annotations

from typing import Any, Callable, Optional
import numpy as np

from jsons.entities import InstanceData
from .schedule_utils import Fixed, Plan, op_options, overlap, plan_cost, validate_plan
from .heuristics import greedy_horizon


# --------------------------------------------------------------------------- windows
def compute_start_windows(instance: InstanceData, horizon: int, fixed: Fixed):
    """(job, k, unit) -> (earliest start, latest start)."""
    windows: dict[tuple[str, int, str], tuple[int, int]] = {}
    for job in instance.jobs:
        opts = [op_options(op) for op in job.operations]
        n = len(opts)
        dmin = [min(o.values()) for o in opts]

        es, prev_end = [0] * n, 0
        for k in range(n):
            es[k] = max(prev_end, fixed.release.get((job.id, k), 0))
            prev_end = es[k] + dmin[k]

        lse, nxt = [0] * n, horizon
        for k in reversed(range(n)):
            lse[k] = min(nxt, fixed.due_end.get((job.id, k), horizon))
            nxt = lse[k] - dmin[k]

        for k in range(n):
            ok = False
            for unit, d in opts[k].items():
                lo, hi = es[k], lse[k] - d
                if hi >= lo:
                    windows[(job.id, k, unit)] = (lo, hi)
                    ok = True
            if not ok:
                raise ValueError(f"Job {job.id}, op {k}: no feasible start window (horizon {horizon})")
    return windows


def calculate_sufficient_lambda(Q_cost: np.ndarray, delta: float = 1.0, safety_margin: float = 1.0) -> float:
    diag = np.diag(Q_cost)
    linear_sum = float(np.sum(np.abs(diag)))
    off = np.triu(Q_cost, k=1) + np.tril(Q_cost, k=-1)
    quad_sum = float(np.sum(np.abs(off))) / 2.0
    return float((linear_sum + quad_sum) / delta + safety_margin)


def calculate_tight_lambda(Q_cost: np.ndarray, factor: float = 2.0) -> float:
    """Practical penalty: factor * largest cost coefficient. Every violation (one-hot, precedence,
    overlap, duty, makespan link) costs >= lambda while it can save at most ~ one coefficient, so this
    is usually enough and keeps the landscape far smoother for annealers than the 'sufficient' bound."""
    return float(max(1.0, factor * np.max(np.abs(np.diag(Q_cost)))))


def _slack_coeffs(X: int, encoding: str = "binary") -> list[int]:
    """Slack in [0, X]. 'binary': log2(X) bits (few variables, rugged landscape for annealers);
    'unary': X bits of weight 1 (more variables, but every +-1 change of the slack is ONE bit flip)."""
    if X <= 0:
        return []
    if encoding == "unary":
        return [1] * X
    nb = X.bit_length()
    return [2 ** b for b in range(nb - 1)] + [X - (2 ** (nb - 1) - 1)]


# --------------------------------------------------------------------------- builder
def build_job_shop_qubo(
    instance: InstanceData,
    horizon: Optional[int] = None,
    penalty_lambda: Optional[float] = None,
    delta: float = 1.0,
    fixed: Optional[Fixed] = None,
    objective: str = "makespan",          # "makespan" | "sum_completion"
    lambda_mode: str = "tight",           # "tight" (practical) | "sufficient" (rigorous, much larger)
    slack_encoding: str = "binary",       # "binary" | "unary"  (duty-cycle slack bits)
) -> tuple[np.ndarray, dict[str, Any]]:
    if objective not in ("makespan", "sum_completion"):
        raise ValueError("objective must be 'makespan' or 'sum_completion'")
    fixed = fixed or Fixed()
    if horizon is None:
        horizon = greedy_horizon(instance, fixed, now=min(fixed.release.values(), default=0))
    windows = compute_start_windows(instance, horizon, fixed)

    # ---- decision variables x
    var_map: dict[tuple[str, int, str, int], int] = {}
    inv_var_map: dict[int, tuple[str, int, str, int]] = {}
    op_vars: dict[tuple[str, int], list[tuple[int, str, int, int]]] = {}        # (idx, unit, t, d)
    unit_vars: dict[str, dict[tuple[str, int], list[tuple[int, int, int]]]] = {}  # (idx, t, d)
    idx = 0
    for job in instance.jobs:
        for k, op in enumerate(job.operations):
            for unit, d in op_options(op).items():
                if (job.id, k, unit) not in windows:
                    continue
                lo, hi = windows[(job.id, k, unit)]
                blocked = fixed.blocked(unit)
                for t in range(lo, hi + 1):
                    if any(overlap(t, t + d, bs, be) for bs, be in blocked):
                        continue
                    var_map[(job.id, k, unit, t)] = idx
                    inv_var_map[idx] = (job.id, k, unit, t)
                    op_vars.setdefault((job.id, k), []).append((idx, unit, t, d))
                    unit_vars.setdefault(unit, {}).setdefault((job.id, k), []).append((idx, t, d))
                    idx += 1
            if (job.id, k) not in op_vars:
                raise ValueError(f"Job {job.id}, op {k}: every start is blocked (breakdown/frozen work)")
    n_decision = idx

    # ---- makespan variables y
    y_map: dict[int, int] = {}
    last_vars: list[tuple[int, int]] = []        # (x index, completion)
    for job in instance.jobs:
        if job.operations:
            k = len(job.operations) - 1
            last_vars += [(i, t + d) for i, _, t, d in op_vars[(job.id, k)]]
    if objective == "makespan" and last_vars:
        t_lo = max(min(t + d for _, _, t, d in op_vars[(j.id, len(j.operations) - 1)])
                   for j in instance.jobs if j.operations)
        t_hi = max(c for _, c in last_vars)
        for T in range(t_lo, t_hi + 1):
            y_map[T] = idx
            idx += 1
    n_y = len(y_map)

    # ---- duty-cycle windows + slack variables
    duty: list[dict[str, Any]] = []
    for unit, (X, Y) in instance.limits.items():
        per_op = unit_vars.get(unit)
        if not per_op:
            continue
        L = X + Y
        frozen = fixed.committed.get(unit, [])
        H_d = max([horizon] + [e for _, e in frozen])
        t_min = min(t for lst in per_op.values() for _, t, _ in lst)
        t_end = max(t + d for lst in per_op.values() for _, t, d in lst)
        s_lo = max(0, t_min - L + 1)
        s_hi = max(s_lo, min(max(0, H_d - L), t_end - 1))
        for s in range(s_lo, s_hi + 1):
            e = min(s + L, H_d)
            x_eff = max(0, X - sum(overlap(a, b, s, e) for a, b in frozen))
            coeffs, max_w = [], 0
            for lst in per_op.values():
                best = 0
                for i, t, d in lst:
                    c = overlap(t, t + d, s, e)
                    if c > 0:
                        coeffs.append((i, c))
                        best = max(best, c)
                max_w += best
            if max_w > x_eff:
                duty.append({"unit": unit, "start": s, "X": x_eff, "coeffs": coeffs})
    slack_map: dict[tuple[str, int], list[int]] = {}
    for w in duty:
        w["slack"] = []
        for a in _slack_coeffs(w["X"], slack_encoding):
            w["slack"].append((idx, a))
            idx += 1
        slack_map[(w["unit"], w["start"])] = [i for i, _ in w["slack"]]
    n_vars = idx

    Q_pen = np.zeros((n_vars, n_vars))
    Q_cost = np.zeros((n_vars, n_vars))
    Q_duty = np.zeros((n_vars, n_vars))
    offset_pen = 0.0
    offset_duty = 0.0

    def one_hot(ids: np.ndarray):
        nonlocal offset_pen
        Q_pen[ids, ids] -= 1.0
        blk = np.ones((len(ids), len(ids)))
        np.fill_diagonal(blk, 0.0)
        Q_pen[np.ix_(ids, ids)] += blk
        offset_pen += 1.0

    # 1. exactly one start per operation
    for lst in op_vars.values():
        one_hot(np.array([i for i, *_ in lst], dtype=int))

    # 2. precedence
    for job in instance.jobs:
        for k in range(len(job.operations) - 1):
            a, b = op_vars[(job.id, k)], op_vars[(job.id, k + 1)]
            va = np.array([i for i, *_ in a]); vb = np.array([i for i, *_ in b])
            ta = np.array([t for _, _, t, _ in a])[:, None]
            da = np.array([d for *_, d in a])[:, None]
            tb = np.array([t for _, _, t, _ in b])[None, :]
            viol = (tb < ta + da).astype(float)
            Q_pen[np.ix_(va, vb)] += 0.5 * viol
            Q_pen[np.ix_(vb, va)] += 0.5 * viol.T

    # 3. no overlap per unit
    for unit, per_op in unit_vars.items():
        keys = list(per_op)
        for i in range(len(keys)):
            A = per_op[keys[i]]
            va = np.array([v for v, _, _ in A])
            ta = np.array([t for _, t, _ in A])[:, None]
            da = np.array([d for *_, d in A])[:, None]
            for j in range(i + 1, len(keys)):
                B = per_op[keys[j]]
                vb = np.array([v for v, _, _ in B])
                tb = np.array([t for _, t, _ in B])[None, :]
                db = np.array([d for *_, d in B])[None, :]
                ov = (np.maximum(ta, tb) < np.minimum(ta + da, tb + db)).astype(float)
                Q_pen[np.ix_(va, vb)] += 0.5 * ov
                Q_pen[np.ix_(vb, va)] += 0.5 * ov.T

    # 4. duty cycle: (sum c_i x_i + sum a_b z_b - X)^2
    for w in duty:
        terms = w["coeffs"] + w["slack"]
        ids = np.array([i for i, _ in terms], dtype=int)
        a = np.array([c for _, c in terms], dtype=float)
        X = float(w["X"])
        Q_duty[ids, ids] += a * a - 2.0 * X * a
        outer = np.outer(a, a)
        np.fill_diagonal(outer, 0.0)
        Q_duty[np.ix_(ids, ids)] += outer
        offset_duty += X * X

    # 5. objective
    if objective == "makespan":
        if y_map:
            one_hot(np.array(list(y_map.values()), dtype=int))
            for T, yi in y_map.items():
                Q_cost[yi, yi] += float(T)
            for i, c in last_vars:
                for T, yi in y_map.items():
                    if c > T:
                        Q_pen[i, yi] += 0.5
                        Q_pen[yi, i] += 0.5
    else:
        for i, c in last_vars:
            Q_cost[i, i] += float(c)

    if penalty_lambda is None:
        penalty_lambda = (calculate_tight_lambda(Q_cost) if lambda_mode == "tight"
                          else calculate_sufficient_lambda(Q_cost, delta=delta, safety_margin=1.0))
    n_x = n_decision + n_y
    Q_core = (Q_cost + penalty_lambda * Q_pen)[:n_x, :n_x].copy()     # everything except duty cycle
    offset_core = penalty_lambda * offset_pen
    Q_pen += Q_duty
    offset_pen += offset_duty
    Q = Q_cost + penalty_lambda * Q_pen
    offset = penalty_lambda * offset_pen

    md = {
        "Q": Q, "offset": offset, "Q_cost": Q_cost, "Q_pen": Q_pen, "penalty_lambda": penalty_lambda,
        "horizon": horizon, "objective": objective, "n_vars": n_vars,
        "n_decision_vars": n_decision, "n_makespan_vars": n_y, "n_slack_vars": n_vars - n_decision - n_y,
        "var_map": var_map, "inv_var_map": inv_var_map, "windows": windows,
        "slack_map": slack_map, "duty_windows": duty, "Q_core": Q_core, "offset_core": offset_core, "n_x": n_x, "n_duty_windows": len(duty), "fixed": fixed, "instance": instance,
    }
    return Q, md


# --------------------------------------------------------------------------- decoding
def decode_bits(bits: np.ndarray, md: dict[str, Any]) -> Optional[Plan]:
    """bits -> Plan (local op indices), or None when some operation has no/several starts."""
    chosen: dict[tuple[str, int], list[tuple[str, int]]] = {}
    for i, (jid, k, u, t) in md["inv_var_map"].items():
        if bits[i]:
            chosen.setdefault((jid, k), []).append((u, t))
    plan: Plan = {}
    for job in md["instance"].jobs:
        for k, op in enumerate(job.operations):
            c = chosen.get((job.id, k), [])
            if len(c) != 1:
                return None
            u, t = c[0]
            plan[(job.id, k)] = (u, t, t + op_options(op)[u])
    return plan


def pick_best_feasible(samples, md: dict[str, Any]):
    """samples: iterable of (bits, energy). Returns (plan, energy, n_feasible) or (None, None, 0)."""
    inst, fixed, obj = md["instance"], md["fixed"], md["objective"]
    best, n_ok = None, 0
    for bits, energy in samples:
        plan = decode_bits(bits, md)
        if plan is None or validate_plan(inst, plan, fixed):
            continue
        n_ok += 1
        key = (plan_cost(plan, inst, obj), energy)
        if best is None or key < best[0]:
            best = (key, plan, energy)
    if best is None:
        return None, None, 0
    return best[1], best[2], n_ok


# --------------------------------------------------------------------------- solver adapter
def make_qubo_solver(sampler: Callable, objective: str = "makespan",
                     penalty_lambda: Optional[float] = None, horizon_extras=(0, 2, 5),
                     lambda_mode: str = "tight", slack_encoding: str = "binary"):
    """solver(sub_instance, fixed) -> (plan | None, info).  `sampler(Q, offset)` -> [(bits, energy)]."""
    def solve(instance: InstanceData, fixed: Optional[Fixed] = None):
        fixed = fixed or Fixed()
        now = min(fixed.release.values(), default=0)
        h0 = greedy_horizon(instance, fixed, now=now)
        info: dict[str, Any] = {}
        for extra in horizon_extras:
            try:
                Q, md = build_job_shop_qubo(instance, horizon=h0 + extra, penalty_lambda=penalty_lambda,
                                            fixed=fixed, objective=objective, lambda_mode=lambda_mode,
                                            slack_encoding=slack_encoding)
            except ValueError as exc:
                info["error"] = str(exc)
                continue
            samples = sampler(Q, md["offset"], md) if getattr(sampler, "wants_md", False) else sampler(Q, md["offset"])
            plan, energy, n_ok = pick_best_feasible(samples, md)
            info.update(n_vars=md["n_vars"], n_slack_vars=md["n_slack_vars"], horizon=md["horizon"],
                        penalty_lambda=md["penalty_lambda"], n_samples=len(samples), n_feasible=n_ok)
            if plan is not None:
                info["energy"] = energy
                return plan, info
        return None, info
    return solve
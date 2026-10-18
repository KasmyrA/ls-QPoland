"""Greedy dispatcher: tight horizon estimate for the QUBO and 'right-shift' baseline."""
from __future__ import annotations

from typing import Optional

try:
    from ..jsons.entities import InstanceData
    from .schedule_utils import Fixed, Plan, op_options, overlap
except ImportError:
    from entities import InstanceData  # type: ignore
    from schedule_utils import Fixed, Plan, op_options, overlap  # type: ignore


def _duty_violation(work, t, d, limit) -> bool:
    X, Y = limit
    L = X + Y
    for s in range(max(0, t - L + 1), t + d):
        e = s + L
        w = overlap(t, t + d, s, e) + sum(overlap(a, b, s, e) for a, b in work)
        if w > X:
            return True
    return False


def _place(unit, ready, d, blocked, work, limits) -> Optional[int]:
    lim = limits.get(unit)
    if lim and d > lim[0]:
        return None
    t = ready
    for _ in range(200000):
        jump = None
        for bs, be in blocked.get(unit, []):
            if t < be and bs < t + d:
                jump = max(jump or 0, be)
        if jump is not None:
            t = jump
            continue
        if lim and _duty_violation(work.get(unit, []), t, d, lim):
            t += 1
            continue
        return t
    return None


def greedy_schedule(instance: InstanceData, fixed: Optional[Fixed] = None, now: int = 0,
                    follow: Optional[dict] = None, rule: str = "ect", jitter: float = 0.0,
                    rng=None) -> Optional[Plan]:
    """Dispatch operations one by one (earliest completion first).

    follow = {(job, k): (unit, min_start)} -> keep the given unit and never start earlier
    (used for the 'right shift' baseline after a breakdown). Ignores due_end.
    """
    fixed = fixed or Fixed()
    limits = instance.limits
    blocked = {u: fixed.blocked(u) for u in set(instance.machines) | set(fixed.outages) | set(fixed.committed)}
    work = {u: list(v) for u, v in fixed.committed.items()}
    plan: Plan = {}
    nxt = {j.id: 0 for j in instance.jobs if j.operations}
    ready = {j.id: max(now, fixed.release.get((j.id, 0), now)) for j in instance.jobs}
    jobs = {j.id: j for j in instance.jobs}
    remaining = {j.id: [min(op_options(op).values()) for op in j.operations] for j in instance.jobs}

    while nxt:
        best = None
        for jid, k in nxt.items():
            op = jobs[jid].operations[k]
            r = max(ready[jid], fixed.release.get((jid, k), 0), now)
            if follow is not None:
                u0, s0 = follow[(jid, k)]
                options = {u0: op_options(op)[u0]}
                r = max(r, s0)
            else:
                options = op_options(op)
            for u, d in options.items():
                t = _place(u, r, d, blocked, work, limits)
                if t is None:
                    continue
                if follow:
                    pri = follow[(jid, k)][1]
                elif rule == "lrpt":                       # most remaining work first
                    pri = t - sum(remaining[jid][k:]) + (rng.uniform(0, jitter) if rng else 0.0)
                else:                                      # earliest completion
                    pri = t + d + (rng.uniform(0, jitter) if rng else 0.0)
                cand = (pri, t + d, jid, u, t, d)
                if best is None or cand[:3] < best[:3]:
                    best = cand
        if best is None:
            return None
        _, _, jid, u, t, d = best
        k = nxt[jid]
        plan[(jid, k)] = (u, t, t + d)
        blocked.setdefault(u, []).append((t, t + d))
        work.setdefault(u, []).append((t, t + d))
        ready[jid] = t + d
        if k + 1 < len(jobs[jid].operations):
            nxt[jid] = k + 1
        else:
            del nxt[jid]
    return plan


def greedy_horizon(instance: InstanceData, fixed: Optional[Fixed] = None, now: int = 0) -> int:
    plan = best_greedy_plan(instance, fixed, now)
    if plan:
        return max(e for _, _, e in plan.values())
    fixed = fixed or Fixed()
    tail = max([e for iv in list(fixed.outages.values()) + list(fixed.committed.values()) for _, e in iv] + [now])
    return tail + sum(max(op_options(op).values()) for j in instance.jobs for op in j.operations) * 2


def best_greedy_plan(instance: InstanceData, fixed: Optional[Fixed] = None, now: int = 0,
                     tries: int = 24, seed: int = 0) -> Optional[Plan]:
    """Best of several dispatch rules / random tie-breaks (cheap, gives a TIGHT horizon)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    best = None
    for i in range(tries):
        rule = "ect" if i % 2 == 0 else "lrpt"
        plan = greedy_schedule(instance, fixed, now, rule=rule, jitter=0.0 if i < 2 else 1.0 + i % 3, rng=rng)
        if plan and (best is None or max(e for _, _, e in plan.values()) < max(e for _, _, e in best.values())):
            best = plan
    return best
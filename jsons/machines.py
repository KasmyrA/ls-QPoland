"""Machine configuration (second JSON), breakdown events (third JSON) and instance expansion.

machines.json
{
  "time_unit": "h",
  "machines": {"T": {"count": 2, "max_work": 6, "min_rest": 3}, "F": {"count": 1}, ...},
  "duration_overrides": {"Z2/1": {"F": 3, "S": 4}}
}
  * count               number of identical parallel units of a type (T -> T1, T2; count 1 keeps the name)
  * max_work / min_rest duty cycle per unit (both required, same time unit as durations)
  * duration_overrides  key "job/op_index" -> {type_or_unit: duration}
        - key equal to the operation's own type  -> changes its duration
        - key of ANOTHER type / unit             -> adds that machine as an ALTERNATIVE with that duration
        - a unit name (e.g. "T2") beats a type name (e.g. "T")

events.json
{
  "now": 4,
  "events": [{"type": "breakdown", "machine": "T2", "start": 4, "duration": 3}]
}
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Union

try:
    from .entities import InstanceData, Job, Operation
except ImportError:
    from entities import InstanceData, Job, Operation  # type: ignore


def _read_json(source: Union[str, Path, dict]) -> dict[str, Any]:
    if isinstance(source, dict):
        return source
    if isinstance(source, (str, Path)):
        with open(source, "r", encoding="utf-8") as f:
            return json.load(f)
    raise TypeError(f"Expected path or dict, got {type(source)}")


def load_machine_config(source: Union[str, Path, dict]) -> dict[str, Any]:
    return _read_json(source)


def load_events(source: Union[str, Path, dict]) -> dict[str, Any]:
    data = _read_json(source)
    for ev in data.get("events", []):
        if ev.get("type", "breakdown") != "breakdown":
            raise ValueError(f"Unsupported event type: {ev.get('type')}")
        if "machine" not in ev or "start" not in ev or not ("duration" in ev or "end" in ev):
            raise ValueError(f"Breakdown needs machine, start and duration/end: {ev}")
    return data


def expand_instance(instance: InstanceData, cfg: dict[str, Any]) -> InstanceData:
    """Expand machine TYPES into physical UNITS and apply limits / duration overrides."""
    mcfg = cfg.get("machines", {})
    overrides = cfg.get("duration_overrides", {})

    types = list(dict.fromkeys(list(instance.machines) + list(mcfg)))
    units_by_type: dict[str, list[str]] = {}
    for typ in types:
        n = int(mcfg.get(typ, {}).get("count", 1))
        if n < 1:
            raise ValueError(f"machines.{typ}.count must be >= 1")
        units_by_type[typ] = [typ] if n == 1 else [f"{typ}{i + 1}" for i in range(n)]

    unit_type: dict[str, str] = {}
    for typ, units in units_by_type.items():
        for u in units:
            if u in unit_type:
                raise ValueError(f"Unit name collision: {u}")
            unit_type[u] = typ

    limits: dict[str, tuple[int, int]] = {}
    for typ, units in units_by_type.items():
        mw, mr = mcfg.get(typ, {}).get("max_work"), mcfg.get(typ, {}).get("min_rest")
        if (mw is None) != (mr is None):
            raise ValueError(f"machines.{typ}: give both max_work and min_rest or neither")
        if mw is not None:
            if int(mw) <= 0 or int(mr) < 0:
                raise ValueError(f"machines.{typ}: need max_work > 0 and min_rest >= 0")
            for u in units:
                limits[u] = (int(mw), int(mr))

    new_jobs: list[Job] = []
    for job in instance.jobs:
        ops: list[Operation] = []
        for k, op in enumerate(job.operations):
            if op.machine not in units_by_type:
                raise ValueError(f"{job.id}/{k}: unknown machine type {op.machine}")
            options = {u: int(op.duration) for u in units_by_type[op.machine]}
            ov = overrides.get(f"{job.id}/{k}", {})
            for key, dur in ov.items():                      # pass 1: type-level
                if key in units_by_type:
                    for u in units_by_type[key]:
                        options[u] = int(dur)
            for key, dur in ov.items():                      # pass 2: unit-level (wins)
                if key not in units_by_type:
                    if key not in unit_type:
                        raise ValueError(f"duration_overrides[{job.id}/{k}]: unknown machine {key}")
                    options[key] = int(dur)
            for u in list(options):                          # drop units that can never run it
                if options[u] <= 0:
                    raise ValueError(f"{job.id}/{k}: duration on {u} must be > 0")
                if u in limits and options[u] > limits[u][0]:
                    warnings.warn(f"{job.id}/{k}: duration {options[u]} > max_work {limits[u][0]} on {u}; unit dropped")
                    del options[u]
            if not options:
                raise ValueError(f"{job.id}/{k}: no machine can execute this operation")
            ops.append(Operation(machine=op.machine, duration=op.duration, step=op.step,
                                 job_id=op.job_id or job.id, machines=options))
        new_jobs.append(Job(id=job.id, operations=ops))

    return InstanceData(
        instance_name=instance.instance_name, problem=instance.problem, jobs=new_jobs,
        machines=list(unit_type), raw_data=instance.raw_data,
        unit_type=unit_type, limits=limits,
    )


def resolve_outages(instance: InstanceData, events: dict[str, Any]):
    """events.json -> (now, {unit: [(start, end)]}). A type name is accepted when it has one unit."""
    units_of: dict[str, list[str]] = {}
    for u, t in instance.unit_type.items():
        units_of.setdefault(t, []).append(u)
    outages: dict[str, list[tuple[int, int]]] = {}
    starts = []
    for ev in events.get("events", []):
        name = ev["machine"]
        if name in instance.unit_type:
            unit = name
        elif name in units_of and len(units_of[name]) == 1:
            unit = units_of[name][0]
        elif name in units_of:
            raise ValueError(f"'{name}' is ambiguous ({units_of[name]}); use a unit name")
        else:
            raise ValueError(f"Unknown machine in event: {name}")
        s = int(ev["start"])
        e = int(ev["end"]) if "end" in ev else s + int(ev["duration"])
        if e <= s:
            raise ValueError(f"Empty outage: {ev}")
        outages.setdefault(unit, []).append((s, e))
        starts.append(s)
    now = int(events["now"]) if "now" in events else (min(starts) if starts else 0)
    return now, outages
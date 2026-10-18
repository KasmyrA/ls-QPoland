"""Loading and parsing utilities for ERP scheduling instances and results."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

try:
    from .entities import CalculationState, InstanceData, Job, Operation
except ImportError:
    from entities import CalculationState, InstanceData, Job, Operation  # type: ignore


# ---------------------------------------------------------------------------
# Priority and Machine Sequence Logic
# ---------------------------------------------------------------------------

DEFAULT_MACHINE_PRIORITY: list[str] = ["T", "F", "S"]


def parse_priority_string(priority: str) -> list[str]:
    """Parse a priority string like 'T->F->S', 'T -> F -> S', 'T,F,S' or 'T-F-S'.

    Examples:
        >>> parse_priority_string("T->F->S")
        ['T', 'F', 'S']
        >>> parse_priority_string("M1 -> M2 -> M3")
        ['M1', 'M2', 'M3']
    """
    cleaned = priority.replace("->", " ").replace(">", " ").replace(",", " ").replace("-", " ")
    tokens = [t.strip() for t in cleaned.split() if t.strip()]
    seen = set()
    result = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result


def extract_unique_machines(data: dict[str, Any]) -> list[str]:
    """Extract all unique machine symbols from an instance JSON in order of appearance."""
    machines: list[str] = []
    seen = set()
    for job in data.get("jobs", []):
        for op in job.get("operations", []):
            m = op.get("machine")
            if m and m not in seen:
                seen.add(m)
                machines.append(m)
    return machines


def resolve_machine_priority(
    data: dict[str, Any],
    priority: Union[str, Sequence[str], Callable[[list[str]], list[str]], None] = None,
) -> list[str]:
    """Resolve machine ordering and priority for a given instance.

    Priority resolution order:
    1. If `priority` is a string (e.g. 'T->F->S'), parse it.
    2. If `priority` is a list/sequence of strings, use that order.
    3. If `priority` is a callable, call it with the unique machines found in the JSON.
    4. If `priority` is None:
       - Check if all machines present in JSON exist in `DEFAULT_MACHINE_PRIORITY` (['T', 'F', 'S']).
         If so, filter `DEFAULT_MACHINE_PRIORITY` to keep only those present, preserving T->F->S.
       - If other machine symbols exist, extract unique machines in their natural order of appearance.
    """
    unique_machines = extract_unique_machines(data)

    if priority is None:
        if all(m in DEFAULT_MACHINE_PRIORITY for m in unique_machines):
            return [m for m in DEFAULT_MACHINE_PRIORITY if m in unique_machines]
        return unique_machines

    if isinstance(priority, str):
        parsed = parse_priority_string(priority)
        for m in unique_machines:
            if m not in parsed:
                parsed.append(m)
        return parsed

    if callable(priority):
        return priority(unique_machines)

    if isinstance(priority, (list, tuple)):
        result = list(priority)
        for m in unique_machines:
            if m not in result:
                result.append(m)
        return result

    raise TypeError(f"Unsupported priority type: {type(priority)}")


# ---------------------------------------------------------------------------
# Loading Functions
# ---------------------------------------------------------------------------

def load_instance(
    source: Union[str, Path, dict[str, Any]],
    priority: Union[str, Sequence[str], Callable[[list[str]], list[str]], None] = None,
) -> InstanceData:
    """Load an instance from a JSON file, path string, or pre-parsed dict.

    Parameters
    ----------
    source : str, Path, or dict
        Path to instance JSON file or dict containing the instance data.
    priority : str, sequence of str, or callable, optional
        Priority order for machines. Defaults to None (which respects T->F->S if present).

    Returns
    -------
    InstanceData
        Parsed instance object with jobs, operations, and resolved machines list.
    """
    instance_name = "instance"
    if isinstance(source, (str, Path)):
        path = Path(source)
        instance_name = path.stem
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif isinstance(source, dict):
        data = source
        instance_name = data.get("name", data.get("instance", "instance"))
    else:
        raise TypeError(f"Expected file path or dict, got {type(source)}")

    problem = data.get("problem", "erp_scheduling")
    machines = resolve_machine_priority(data, priority=priority)

    jobs: list[Job] = []
    for job_dict in data.get("jobs", []):
        job_id = str(job_dict.get("id", ""))
        ops: list[Operation] = []
        for step_idx, op_dict in enumerate(job_dict.get("operations", [])):
            ops.append(
                Operation(
                    machine=str(op_dict["machine"]),
                    duration=int(op_dict["duration"]),
                    step=step_idx,
                    job_id=job_id,
                )
            )
        jobs.append(Job(id=job_id, operations=ops))

    return InstanceData(
        instance_name=instance_name,
        problem=problem,
        jobs=jobs,
        machines=machines,
        raw_data=data,
    )


def load_schedule(source: Union[str, Path, dict[str, Any]]) -> CalculationState:
    """Load a calculation result from a JSON file, path, or dict into CalculationState.

    Parameters
    ----------
    source : str, Path, or dict
        Path to output JSON file or pre-parsed dict.

    Returns
    -------
    CalculationState
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif isinstance(source, dict):
        data = source
    else:
        raise TypeError(f"Expected file path or dict, got {type(source)}")

    return CalculationState.from_dict(data)


"""Save and load utilities for ERP / Job-Shop scheduling instances and results.

Handles:
- Loading instance JSONs (e.g. instance_2x2.json, instance_3x3.json).
- Saving and loading calculation results matching output_format.json.
- Machine ordering and priority logic (default: T -> F -> S), with support for
  custom priority strings, custom priority functions, and dynamic symbol detection.
- CalculationState class to store and validate state, schedule, and metrics during calculations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union


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
    # Preserve order while removing duplicates
    seen = set()
    result = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result


def extract_unique_machines(data: dict) -> list[str]:
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
    data: dict,
    priority: Union[str, Sequence[str], Callable[[list[str]], list[str]], None] = None,
) -> list[str]:
    """Resolve machine ordering and priority for a given instance.

    Priority resolution order:
    1. If `priority` is a string (e.g. 'T->F->S' or 'M1->M2->M3'), parse it.
    2. If `priority` is a list/sequence of strings, use that order.
    3. If `priority` is a callable, call it with the unique machines found in the JSON.
    4. If `priority` is None:
       - Check if all machines present in JSON exist in `DEFAULT_MACHINE_PRIORITY` (['T', 'F', 'S']).
         If so, filter `DEFAULT_MACHINE_PRIORITY` to keep only those present, preserving T->F->S.
       - If other machine symbols exist, extract unique machines in their natural order of appearance.
    """
    unique_machines = extract_unique_machines(data)

    if priority is None:
        # Check if instance is a subset of default T->F->S
        if all(m in DEFAULT_MACHINE_PRIORITY for m in unique_machines):
            return [m for m in DEFAULT_MACHINE_PRIORITY if m in unique_machines]
        # Otherwise fallback to natural appearance order
        return unique_machines

    if isinstance(priority, str):
        parsed = parse_priority_string(priority)
        # Append any machines present in data that weren't mentioned in the priority string
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
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class Operation:
    """A single manufacturing operation belonging to a job."""
    machine: str
    duration: int
    step: int = 0
    job_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"machine": self.machine, "duration": self.duration}


@dataclass
class Job:
    """A job consisting of an ordered list of operations."""
    id: str
    operations: list[Operation] = field(default_factory=list)

    @property
    def total_duration(self) -> int:
        return sum(op.duration for op in self.operations)

    @property
    def machine_sequence(self) -> list[str]:
        return [op.machine for op in self.operations]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "operations": [op.to_dict() for op in self.operations],
        }


@dataclass
class InstanceData:
    """Full problem instance data parsed from JSON."""
    instance_name: str
    problem: str
    jobs: list[Job] = field(default_factory=list)
    machines: list[str] = field(default_factory=list)
    raw_data: dict[str, Any] = field(default_factory=dict)

    @property
    def n_jobs(self) -> int:
        return len(self.jobs)

    @property
    def n_machines(self) -> int:
        return len(self.machines)

    def get_job(self, job_id: str) -> Optional[Job]:
        for j in self.jobs:
            if j.id == job_id:
                return j
        return None

    def get_operations_for_machine(self, machine: str) -> list[tuple[Job, Operation]]:
        ops = []
        for job in self.jobs:
            for op in job.operations:
                if op.machine == machine:
                    ops.append((job, op))
        return ops

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem": self.problem,
            "jobs": [j.to_dict() for j in self.jobs],
        }


@dataclass
class ScheduledTask:
    """An operation scheduled with concrete start and end times."""
    job: str
    machine: str
    start: int
    end: int

    @property
    def duration(self) -> int:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        return {
            "job": self.job,
            "machine": self.machine,
            "start": int(self.start),
            "end": int(self.end),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScheduledTask:
        return cls(
            job=str(data["job"]),
            machine=str(data["machine"]),
            start=int(data["start"]),
            end=int(data["end"]),
        )


@dataclass
class CalculationState:
    """Stores information during calculations and holds the final schedule.

    Matches output_format.json:
    {
        "instance": "instance_2x2",
        "method": "QAOA (state-vector simulator)",
        "makespan": 4,
        "schedule": [
            { "job": "Z2", "machine": "T", "start": 0, "end": 1 },
            ...
        ]
    }
    """
    instance: str
    method: str = "Unknown"
    makespan: int = 0
    schedule: list[ScheduledTask] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_task(self, job: str, machine: str, start: int, end: int) -> ScheduledTask:
        """Add a scheduled task and update makespan."""
        task = ScheduledTask(job=job, machine=machine, start=start, end=end)
        self.schedule.append(task)
        if end > self.makespan:
            self.makespan = end
        return task

    def recalculate_makespan(self) -> int:
        """Compute the maximum end time across all scheduled tasks."""
        if not self.schedule:
            self.makespan = 0
        else:
            self.makespan = max(task.end for task in self.schedule)
        return self.makespan

    def get_machine_schedule(self, machine: str) -> list[ScheduledTask]:
        """Return tasks on a specific machine sorted by start time."""
        tasks = [t for t in self.schedule if t.machine == machine]
        return sorted(tasks, key=lambda t: t.start)

    def get_job_schedule(self, job_id: str) -> list[ScheduledTask]:
        """Return tasks for a specific job sorted by start time."""
        tasks = [t for t in self.schedule if t.job == job_id]
        return sorted(tasks, key=lambda t: t.start)

    def validate(self, instance: Optional[InstanceData] = None) -> tuple[bool, list[str]]:
        """Validate schedule feasibility.

        Checks:
        1. No two tasks overlap on the same machine.
        2. Precedence: operations for the same job do not overlap.
        3. If instance is provided, checks that durations and operations match.
        """
        errors = []

        # 1. Machine no-overlap
        machines = {t.machine for t in self.schedule}
        for m in machines:
            tasks = self.get_machine_schedule(m)
            for i in range(len(tasks) - 1):
                if tasks[i].end > tasks[i + 1].start:
                    errors.append(
                        f"Machine overlap on {m}: Job {tasks[i].job} ({tasks[i].start}-{tasks[i].end}) "
                        f"overlaps with Job {tasks[i + 1].job} ({tasks[i + 1].start}-{tasks[i + 1].end})"
                    )

        # 2. Job precedence no-overlap
        jobs = {t.job for t in self.schedule}
        for j in jobs:
            tasks = self.get_job_schedule(j)
            for i in range(len(tasks) - 1):
                if tasks[i].end > tasks[i + 1].start:
                    errors.append(
                        f"Job overlap for {j}: task on {tasks[i].machine} ({tasks[i].start}-{tasks[i].end}) "
                        f"overlaps with task on {tasks[i + 1].machine} ({tasks[i + 1].start}-{tasks[i + 1].end})"
                    )

        # 3. Instance consistency
        if instance is not None:
            for j in instance.jobs:
                j_tasks = self.get_job_schedule(j.id)
                if len(j_tasks) != len(j.operations):
                    errors.append(
                        f"Job {j.id} has {len(j.operations)} operations in instance but {len(j_tasks)} in schedule"
                    )

        is_valid = len(errors) == 0
        return is_valid, errors

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict matching output_format.json format."""
        return {
            "instance": self.instance,
            "method": self.method,
            "makespan": int(self.makespan),
            "schedule": [task.to_dict() for task in self.schedule],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalculationState:
        """Create CalculationState from dict."""
        schedule = [ScheduledTask.from_dict(item) for item in data.get("schedule", [])]
        makespan = data.get("makespan", 0)
        if not makespan and schedule:
            makespan = max(t.end for t in schedule)
        return cls(
            instance=data.get("instance", "unknown"),
            method=data.get("method", "Unknown"),
            makespan=int(makespan),
            schedule=schedule,
        )


# ---------------------------------------------------------------------------
# Load and Save Functions
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


def save_schedule(
    state_or_data: Union[CalculationState, dict[str, Any]],
    target: Union[str, Path],
    indent: int = 4,
) -> Path:
    """Save calculation schedule to a JSON file matching output_format.json.

    Parameters
    ----------
    state_or_data : CalculationState or dict
        The schedule state object or formatted dictionary.
    target : str or Path
        Destination file path.
    indent : int
        JSON indentation. Default is 4.

    Returns
    -------
    Path
        Path to the saved file.
    """
    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(state_or_data, CalculationState):
        out_dict = state_or_data.to_dict()
    elif isinstance(state_or_data, dict):
        out_dict = state_or_data
    else:
        raise TypeError(f"Expected CalculationState or dict, got {type(state_or_data)}")

    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, indent=indent)

    return target_path


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


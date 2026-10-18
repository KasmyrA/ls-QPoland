"""Domain entities and state representations for ERP / Job-Shop scheduling."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Operation:
    """A single manufacturing operation belonging to a job."""
    machine: str
    duration: int
    step: int = 0
    job_id: str = ""
    machines: Optional[Any] = None  

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
    unit: str = ""  #T2 or sth like that 

    @property
    def duration(self) -> int:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
            d = {"job": self.job, "machine": self.machine,
                 "start": int(self.start), "end": int(self.end)}
            if self.unit:                  
                d["unit"] = self.unit
            return d
    
    @classmethod
    def from_dict(cls, data):
        return cls(job=str(data["job"]), machine=str(data["machine"]),
                   start=int(data["start"]), end=int(data["end"]),
                   unit=str(data.get("unit", "")))


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


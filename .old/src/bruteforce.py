"""Brute-force scheduler: enumerate all valid job-machine assignments."""
from __future__ import annotations
import itertools
import numpy as np


def enumerate_assignments(n_jobs: int, n_machines: int):
    """Yield every mapping of jobs -> machines (1-hot encoding).

    Each assignment is a length `n_jobs` list giving the machine index per job.
    We additionally yield a 2-D one-hot matrix [jobs x machines].
    """
    for choice in itertools.product(range(n_machines), repeat=n_jobs):
        bitstring = np.zeros((n_jobs, n_machines), dtype=int)
        for j, m in enumerate(choice):
            bitstring[j, m] = 1
        yield list(choice), bitstring


def compute_completion_times(
    durations: np.ndarray, assignment: np.ndarray
) -> tuple[float, dict]:
    """Compute makespan and per-machine completion times.

    Parameters
    ----------
    durations : (n_jobs,) array of processing times.
    assignment  : (n_jobs, n_machines) one-hot matrix from enumerate_assignments.

    Returns
    -------
    makespan : float
        The maximum total time any machine is busy (makespan).
    machine_loads : dict[str, list[float]]
        Per-machine list of job durations processed in order.
    """
    n_jobs, n_machines = assignment.shape
    machine_loads: dict[int, list[float]] = {m: [] for m in range(n_machines)}

    # Group jobs by their assigned machine, preserving job index order
    for j in range(n_jobs):
        row = assignment[j]
        if row.sum() != 1:
            raise ValueError(f"Job {j} not assigned to exactly one machine")
        m = int(np.argmax(row))
        machine_loads[m].append(durations[j])

    loads = {m: sum(machine_loads[m]) for m in range(n_machines)}
    makespan = max(loads.values()) if loads else 0.0
    return makespan, loads


def solve_bruteforce(
    durations: np.ndarray,
    penalty_weight: float | None = None,
    n_machines: int | None = None,
) -> dict:
    """Enumerate all assignments and return the optimum.

    Parameters
    ----------
    durations : (n_jobs,) array of processing times.
    penalty_weight : optional weight for penalty terms (unused here).
    n_machines : number of machines; defaults to len(durations) if None.

    Returns a dictionary with:
      - optimal_assignment: best one-hot matrix
      - optimal_makespan:   minimum makespan
      - energy_table:       list of (makespan, bitstring_flat, valid) tuples
    """
    n_jobs = durations.shape[0]
    if n_machines is None:
        # assume square for backward compatibility
        n_machines = n_jobs

    best_makespan = float("inf")
    best_assignment = None
    energy_table: list[tuple[float, np.ndarray, bool]] = []

    for choice, bitstring in enumerate_assignments(n_jobs, n_machines):
        # Feasibility check
        row_sums = bitstring.sum(axis=1)  # jobs x 1
        col_sums = bitstring.sum(axis=0)  # machines x 1 (no constraint here)

        valid = bool(np.all(row_sums == 1))
        if valid:
            makespan, _ = compute_completion_times(durations, bitstring)
        else:
            makespan = float("inf")

        energy_table.append((makespan, bitstring.flatten(), valid))

        if valid and makespan < best_makespan:
            best_makespan = makespan
            best_assignment = bitstring.copy()

    return {
        "optimal_assignment": best_assignment,
        "optimal_makespan": float(best_makespan),
        "energy_table": energy_table,
        "n_feasible": sum(1 for _, _, v in energy_table if v),
        "n_total": len(energy_table),
    }


def greedy_schedule(durations: np.ndarray) -> tuple[np.ndarray, float]:
    """Classical longest-processing-time (LPT) greedy scheduler.

    Assigns jobs one by one to the machine with the smallest current load.
    Returns (assignment_matrix, makespan).
    """
    n_jobs = durations.shape[0]
    n_machines = n_jobs
    loads = np.zeros(n_machines, dtype=float)
    assignment = np.zeros((n_jobs, n_machines), dtype=int)

    order = np.argsort(-durations)  # descending duration (largest first)

    for j in order:
        m = int(np.argmin(loads))
        assignment[j, m] = 1
        loads[m] += durations[j]

    return assignment, float(loads.max())

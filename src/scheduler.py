"""Classical greedy scheduler: build a Gantt chart from job→machine assignments."""
from __future__ import annotations
import numpy as np


def schedule_from_assignment(
    durations: np.ndarray,
    assignment: list[int],
) -> dict:
    """Construct a schedule given a flat job→machine mapping.

    Parameters
    ----------
    durations : (n_jobs,) array of processing times.
    assignment  : length n_jobs list; assignment[j] = machine index for job j, or -1 if unassigned.

    Returns
    -------
    dict with:
        makespan     : float — max completion time across machines
        machine_loads: dict[int, list[float]] — durations per machine in order
        gantt        : dict[int, list[tuple[int,int,int]]] — (job_idx, start, end) per machine
    """
    n_jobs = len(durations)
    n_machines = max(assignment) + 1 if assignment else 0

    # Group jobs by machine in original job order
    machine_loads: dict[int, list[float]] = {m: [] for m in range(n_machines)}
    machine_job_indices: dict[int, list[int]] = {m: [] for m in range(n_machines)}

    for j in range(n_jobs):
        if assignment[j] < 0 or assignment[j] >= n_machines:
            continue
        m = assignment[j]
        machine_loads[m].append(float(durations[j]))
        machine_job_indices[m].append(j)

    # Compute start/end times per machine (sequential on each machine)
    gantt: dict[int, list[tuple[int, int, int]]] = {}
    for m in range(n_machines):
        starts_ends = []
        t = 0.0
        for dur in machine_loads[m]:
            s = t
            e = t + dur
            g_idx = machine_job_indices[m][len(starts_ends)]
            starts_ends.append((g_idx, int(s), int(e)))
            t = e
        gantt[m] = starts_ends

    makespan = max((sum(machine_loads[m]) for m in range(n_machines)), default=0.0)

    return {
        "makespan": float(makespan),
        "machine_loads": machine_loads,
        "gantt": gantt,
        "assignment": assignment,
        "durations": durations.tolist(),
    }


def greedy_lpt(durations: np.ndarray) -> dict:
    """Longest-Processing-Time first greedy scheduler.

    Assigns the largest remaining job to the currently least-loaded machine.
    Returns a schedule dict (same format as schedule_from_assignment).
    """
    n_jobs = len(durations)
    # Sort jobs by descending duration; track original indices
    order = np.argsort(-durations)
    assignment = [-1] * n_jobs
    loads = [0.0] * n_jobs

    for j_orig in order:
        m = int(np.argmin(loads))
        assignment[j_orig] = m
        loads[m] += durations[j_orig]

    return schedule_from_assignment(durations, assignment)


def generate_gantt_html(schedule: dict) -> str:
    """Render a Gantt chart as an inline HTML table."""
    gantt = schedule["gantt"]
    makespan = schedule["makespan"]
    rows = []
    for m in sorted(gantt.keys()):
        for job_idx, start, end in gantt[m]:
            pct_start = (start / makespan) * 100 if makespan > 0 else 0
            width_pct = ((end - start) / makespan) * 100 if makespan > 0 else 0
            rows.append(
                f'<tr><td>M{m}</td>'
                f'<td>Job {job_idx} [{int(schedule["durations"][job_idx])}]</td>'
                f'<td style="width:{pct_start:.1f}%"></td>'
                f'<td style="width:{width_pct:.1f}%;background:#4a90d9;color:white;text-align:center;">'
                f'{start}–{end}</td></tr>'
            )

    html = (
        "<table border=1 cellpadding=4 cellspacing=0>"
        "<tr><th>Machine</th><th>Job</th><th colspan=2>Gantt</th></tr>"
        + "\n".join(rows) + "</table>"
    )
    return html


def generate_gantt_svg(schedule: dict, width=600, bar_height=30) -> str:
    """Render a Gantt chart as SVG."""
    gantt = schedule["gantt"]
    makespan = schedule["makespan"] or 1
    n_machines = len(gantt)

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{n_machines * bar_height + 40}">',
        '<rect width="100%" height="100%" fill="#f8f9fa"/>',
    ]

    for m in sorted(gantt.keys()):
        y = m * bar_height + 20
        # Machine label
        svg_parts.append(
            f'<text x="5" y="{y + bar_height // 2}" '
            f'font-family="monospace" font-size="14" dominant-baseline="middle">M{m}</text>'
        )
        for job_idx, start, end in gantt[m]:
            x = (start / makespan) * (width - 60) + 50
            w = ((end - start) / makespan) * (width - 60)
            svg_parts.append(
                f'<rect x="{x}" y="{y + 5}" width="{w:.1f}" height="{bar_height - 10}" '
                f'fill="#4a90d9" rx="3"/>'
            )
            dur = schedule["durations"][job_idx]
            svg_parts.append(
                f'<text x="{x + w // 2}" y="{y + bar_height // 2}" '
                f'font-family="monospace" font-size="11" text-anchor="middle" fill="white">'
                f'J{job_idx} [{int(dur)}]</text>'
            )

    svg_parts.append("</svg>")
    return "\n".join(svg_parts)

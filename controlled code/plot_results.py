"""Create Gantt-chart images from Job-Shop scheduling result JSON files.

Examples:
    python "controlled code/plot_results.py"
    python "controlled code/plot_results.py" "controlled code/resources/result_instance_2x2.json"
    python "controlled code/plot_results.py" --output-dir /tmp/plots
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# The project may be run on machines where the home configuration directory is
# read-only.  Keep Matplotlib's cache in a writable temporary location.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "quantum-scheduling-mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = PROJECT_DIR / "resources"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_DIR / "plots"


def load_result(path: Path) -> dict[str, Any]:
    """Load and minimally validate one result JSON file."""
    with path.open(encoding="utf-8") as source:
        result = json.load(source)

    if not isinstance(result, dict):
        raise ValueError("The result JSON must be an object.")
    if not isinstance(result.get("schedule"), list) or not result["schedule"]:
        raise ValueError("The result JSON must contain a non-empty 'schedule' list.")

    for index, task in enumerate(result["schedule"]):
        if not isinstance(task, dict):
            raise ValueError(f"Schedule item {index} must be an object.")
        missing = {"job", "machine", "start", "end"} - task.keys()
        if missing:
            raise ValueError(f"Schedule item {index} misses: {', '.join(sorted(missing))}.")
        if int(task["end"]) < int(task["start"]):
            raise ValueError(f"Schedule item {index} has an end before its start.")

    return result


def unique_in_order(values: Iterable[str]) -> list[str]:
    """Return unique values without changing their first-occurrence order."""
    return list(dict.fromkeys(values))


def plot_result(result: dict[str, Any], output_path: Path) -> Path:
    """Render one scheduling result as a machine-oriented Gantt chart."""
    schedule = result["schedule"]
    machines = unique_in_order(str(task["machine"]) for task in schedule)
    jobs = unique_in_order(str(task["job"]) for task in schedule)
    max_end = max(int(task["end"]) for task in schedule)
    makespan = int(result.get("makespan", max_end))
    timeline_end = max(max_end, makespan, 1)

    color_map = {
        job: plt.get_cmap("tab20")(index % 20)
        for index, job in enumerate(jobs)
    }
    machine_positions = {machine: index for index, machine in enumerate(machines)}

    figure, axis = plt.subplots(figsize=(max(8, timeline_end * 1.15), max(3.5, len(machines) * 1.15)))
    for task in schedule:
        job = str(task["job"])
        machine = str(task["machine"])
        start = int(task["start"])
        end = int(task["end"])
        duration = end - start
        position = machine_positions[machine]

        axis.barh(
            position,
            duration,
            left=start,
            height=0.58,
            color=color_map[job],
            edgecolor="black",
            linewidth=0.8,
        )
        if duration:
            axis.text(
                start + duration / 2,
                position,
                f"{job}\n{start}–{end}",
                ha="center",
                va="center",
                fontsize=9,
            )

    axis.axvline(makespan, color="#c62828", linestyle="--", linewidth=1.5, label="Makespan")
    axis.set_yticks(range(len(machines)), machines)
    axis.invert_yaxis()
    axis.set_xlim(0, timeline_end + 0.35)
    axis.set_xticks(range(timeline_end + 1))
    axis.set_xlabel("Czas")
    axis.set_ylabel("Maszyna")
    axis.grid(axis="x", color="#bdbdbd", alpha=0.55, linewidth=0.7)
    axis.set_axisbelow(True)

    instance = result.get("instance", output_path.stem)
    method = result.get("method", "Unknown method")
    axis.set_title(f"Harmonogram: {instance}\n{method} · makespan = {makespan}")
    legend_handles = [Patch(facecolor=color_map[job], edgecolor="black", label=job) for job in jobs]
    legend_handles.append(Patch(facecolor="none", edgecolor="#c62828", label="Makespan"))
    axis.legend(handles=legend_handles, title="Zadania", loc="upper left", bbox_to_anchor=(1.01, 1))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Gantt charts from scheduling results.")
    parser.add_argument(
        "results",
        nargs="*",
        type=Path,
        help="Result JSON files. If omitted, all resources/result_*.json files are used.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for PNG files (default: {DEFAULT_OUTPUT_DIR}).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    paths = args.results or sorted(DEFAULT_RESULTS_DIR.glob("result_*.json"))
    if not paths:
        print("No result JSON files found.")
        return 1

    errors = 0
    for path in paths:
        try:
            output_path = args.output_dir / f"{path.stem}.png"
            plot_result(load_result(path), output_path)
            print(f"Created: {output_path}")
        except (OSError, ValueError, json.JSONDecodeError) as error:
            errors += 1
            print(f"Could not process {path}: {error}")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

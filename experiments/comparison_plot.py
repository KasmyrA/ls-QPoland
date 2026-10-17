"""Comparison plot: makespan vs penalty_weight for different instances."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_instance(path: Path) -> dict:
    with open(path) as f:
        data = json.load(f)
    return {
        "name": path.stem,
        "durations": np.array(data["durations"]),
        "n_jobs": data["n_jobs"],
        "n_machines": data["n_machines"],
    }


def main():
    instance_dir = Path(__file__).parent.parent / "instance"
    results_dir = Path(__file__).parent.parent / "results"

    instances = [load_instance(p) for p in sorted(instance_dir.glob("*.json"))]

    fig, axes = plt.subplots(1, len(instances), figsize=(5 * len(instances), 4))
    if len(instances) == 1:
        axes = [axes]

    colors = ["#2E86AB", "#A23B72", "#F18F01"]

    for idx, inst in enumerate(instances):
        ax = axes[idx]
        durations = inst["durations"]
        n_jobs, n_machines = inst["n_jobs"], inst["n_machines"]

        # Brute-force makespan (baseline)
        from src.bruteforce import solve_bruteforce

        brute = solve_bruteforce(durations, n_machines=n_machines)
        optimal_makespan = brute["optimal_makespan"]

        # Annealing with different penalty weights
        from src.anneal import anneal_multi

        penalties = [0.1, 1.0, 10.0, 100.0, 1000.0]
        ann_energies = []
        ann_feasibilities = []

        for p in penalties:
            ann = anneal_multi(
                durations,
                seeds=[0],
                num_reads=200,
                penalty_weight=p,
            )
            ann_energies.append(ann["best_overall_energy"] or float("inf"))
            ann_feasibilities.append(ann["feasibility_rate"])

        # Plot
        ax.scatter(penalties, ann_energies, color=colors[idx], s=80, zorder=5)
        ax.plot(penalties, ann_energies, "--", alpha=0.3, color=colors[idx])
        ax.axhline(
            optimal_makespan,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"Optimal: {optimal_makespan}",
        )
        ax.set_xlabel("Penalty weight")
        ax.set_ylabel("Best annealing energy")
        ax.set_title(
            f"{inst['name']} ({n_jobs}×{n_machines})", fontweight="bold"
        )
        ax.grid(True, alpha=0.3)

        # Feasibility rate (secondary axis)
        ax2 = ax.twinx()
        ax2.bar(
            penalties,
            ann_feasibilities,
            width=0.6,
            alpha=0.15,
            color="red",
            label="Feasibility",
        )
        ax2.set_ylabel("Feasibility rate")

    plt.tight_layout()
    out_path = results_dir / "comparison_makespan_vs_penalty.png"
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"Saved comparison plot to {out_path}")


if __name__ == "__main__":
    main()

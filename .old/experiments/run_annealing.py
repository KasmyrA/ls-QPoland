"""Run simulated annealing on all instances and save results."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.anneal import anneal_multi, exact_enumeration
from src.bruteforce import solve_bruteforce


def load_instance(path: Path) -> dict:
    with open(path) as f:
        data = json.load(f)
    return {
        "name": path.stem,
        "durations": np.array(data["durations"]),
        "n_jobs": data["n_jobs"],
        "n_machines": data["n_machines"],
        "description": data.get("description", ""),
    }


def main():
    instance_dir = Path(__file__).parent.parent / "instance"
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    for inst_path in sorted(instance_dir.glob("*.json")):
        print(f"\n=== {inst_path.stem} ===")
        inst = load_instance(inst_path)
        durations = inst["durations"]

        # Brute-force baseline (with correct n_machines)
        print("  bruteforce...")
        brute = solve_bruteforce(durations, n_machines=inst["n_machines"])
        optimal_makespan = brute["optimal_makespan"]
        n_feasible = brute["n_feasible"]

        # Annealing
        print("  anneal_multi (3 seeds)...")
        ann = anneal_multi(
            durations,
            seeds=[0, 1, 2],
            num_reads=500,
            penalty_weight=None,
        )

        best_ann_energy = ann["best_overall_energy"]
        feasibility_rate = ann["feasibility_rate"]

        # Exact enumeration (small instances only)
        n_bits = durations.shape[0] ** 2
        if n_bits <= 9:  # keep exact feasible
            print("  exact_enumeration...")
            exact = exact_enumeration(durations, penalty_weight=None)
            exact_feasible_count = sum(1 for r in exact["all_samples"] if r["valid"])
        else:
            exact = None
            exact_feasible_count = None

        # Save
        out = {
            "instance": inst["name"],
            "n_jobs": inst["n_jobs"],
            "n_machines": inst["n_machines"],
            "durations": durations.tolist(),
            "optimal_makespan": optimal_makespan,
            "n_feasible_assignments": n_feasible,
            "anneal": {
                "best_energy": best_ann_energy,
                "feasibility_rate": feasibility_rate,
                "num_reads_per_seed": 500,
            },
            "exact_enumeration": {
                "n_total_samples": exact["n_total"] if exact else None,
                "n_feasible": exact_feasible_count,
                "elapsed_s": exact["elapsed_s"] if exact else None,
            },
        }

        out_path = results_dir / f"annealing_{inst['name']}.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  → {out_path}")


if __name__ == "__main__":
    main()

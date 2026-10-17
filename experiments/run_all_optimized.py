"""Optimized experiment runner — skips brute-force for large instances."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.anneal import anneal_multi


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
    results_dir.mkdir(exist_ok=True)

    for inst_path in sorted(instance_dir.glob("*.json")):
        print(f"\n=== {inst_path.stem} ===")
        inst = load_instance(inst_path)
        durations = inst["durations"]
        n_jobs, n_machines = inst["n_jobs"], inst["n_machines"]

        # Brute-force only for small instances (fast)
        if len(durations) <= 5:
            from src.bruteforce import solve_bruteforce
            print("  brute...")
            brute = solve_bruteforce(durations, n_machines=inst["n_machines"])
            optimal_makespan = brute["optimal_makespan"]
            n_feasible = brute["n_feasible"]
            print(f"  → makespan: {optimal_makespan}, feasible: {n_feasible}")
        else:
            optimal_makespan = None
            n_feasible = None
            print("  (brute-force skipped for large instances)")

        # Annealing — reduced reads for medium/large
        num_reads = 100 if len(durations) > 5 else 200
        print(f"  anneal ({num_reads} reads)...")
        ann = anneal_multi(
            durations,
            seeds=[0],
            num_reads=num_reads,
            penalty_weight=None,
        )
        best_ann_energy = ann["best_overall_energy"]
        feasibility_rate = ann["feasibility_rate"]
        print(f"  → energy: {best_ann_energy:.3f}, feasibility: {feasibility_rate:.1%}")

        # Save results
        out = {
            "instance": inst["name"],
            "n_jobs": n_jobs,
            "n_machines": n_machines,
            "durations": durations.tolist(),
            "optimal_makespan": optimal_makespan,
            "n_feasible_assignments": n_feasible,
            "anneal": {
                "best_energy": best_ann_energy,
                "feasibility_rate": feasibility_rate,
                "num_reads_per_seed": num_reads,
            },
        }

        out_path = results_dir / f"annealing_{inst['name']}.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  → saved to {out_path}")


if __name__ == "__main__":
    main()

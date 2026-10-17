"""Run experiments on all instances with reduced scope for medium/large."""
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

        # Brute-force baseline
        print("  bruteforce...")
        brute = solve_bruteforce(durations)
        optimal_makespan = brute["optimal_makespan"]
        n_feasible = brute["n_feasible"]
        print(f"  → optimal makespan: {optimal_makespan}, feasible: {n_feasible}")

        # Annealing — reduced reads for medium/large
        num_reads = 200 if len(durations) > 5 else 500
        print(f"  anneal_multi ({num_reads} reads)...")
        ann = anneal_multi(
            durations,
            seeds=[0, 1],
            num_reads=num_reads,
            penalty_weight=None,
        )
        best_ann_energy = ann["best_overall_energy"]
        feasibility_rate = ann["feasibility_rate"]
        print(f"  → energy: {best_ann_energy:.3f}, feasibility: {feasibility_rate:.1%}")

        # Exact enumeration (small only)
        n_bits = durations.shape[0] ** 2
        exact = None
        if n_bits <= 9:
            print("  exact_enumeration...")
            exact = exact_enumeration(durations, penalty_weight=None)
            exact_feasible_count = sum(1 for r in exact["all_samples"] if r["valid"])
            print(f"  → {exact_feasible_count}/{exact['n_total']} feasible, elapsed: {exact['elapsed_s']:.3f}s")

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
            "exact_enumeration": {
                "n_total_samples": exact["n_total"] if exact else None,
                "n_feasible": sum(1 for r in (exact["all_samples"] if exact else [])) if exact else None,
                "elapsed_s": exact["elapsed_s"] if exact else None,
            },
        }

        out_path = results_dir / f"annealing_{inst['name']}.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  → saved to {out_path}")


if __name__ == "__main__":
    main()

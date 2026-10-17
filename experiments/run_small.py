"""Run annealing on small instance only (fast validation)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.anneal import anneal_multi, exact_enumeration
from src.bruteforce import solve_bruteforce


def main():
    inst_path = Path(__file__).parent.parent / "instance" / "small.json"
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    with open(inst_path) as f:
        data = json.load(f)
    durations = np.array(data["durations"])

    print("=== small.json (3×3) ===")
    print(f"  durations: {durations}")

    # Brute-force baseline
    print("  bruteforce...")
    brute = solve_bruteforce(durations)
    optimal_makespan = brute["optimal_makespan"]
    n_feasible = brute["n_feasible"]
    print(f"  → optimal makespan: {optimal_makespan}, feasible assignments: {n_feasible}")

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
    print(f"  → annealing: energy={best_ann_energy:.3f}, feasibility={feasibility_rate:.1%}")

    # Exact enumeration
    print("  exact_enumeration...")
    exact = exact_enumeration(durations, penalty_weight=None)
    exact_feasible_count = sum(1 for r in exact["all_samples"] if r["valid"])
    n_total = exact["n_total"]
    elapsed_s = exact["elapsed_s"]
    print(f"  → {exact_feasible_count}/{n_total} feasible, elapsed: {elapsed_s:.2f}s")

    # Save results
    out = {
        "instance": "small",
        "n_jobs": 3,
        "n_machines": 3,
        "durations": durations.tolist(),
        "optimal_makespan": optimal_makespan,
        "n_feasible_assignments": n_feasible,
        "anneal": {
            "best_energy": best_ann_energy,
            "feasibility_rate": feasibility_rate,
            "num_reads_per_seed": 500,
        },
        "exact_enumeration": {
            "n_total_samples": n_total,
            "n_feasible": exact_feasible_count,
            "elapsed_s": elapsed_s,
        },
    }

    out_path = results_dir / "annealing_small.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  → saved to {out_path}")


if __name__ == "__main__":
    main()

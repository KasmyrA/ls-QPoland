"""End-to-end integration: instance → brute-force → anneal → compare."""
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
        "durations": np.array(data["durations"]),
        "n_jobs": data["n_jobs"],
        "n_machines": data["n_machines"],
    }


def main():
    instance_dir = Path(__file__).parent.parent / "instance"
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Use small.json (fast validation)
    inst_path = instance_dir / "small.json"
    print(f"=== Integration test: {inst_path.name} ===")

    inst = load_instance(inst_path)
    durations = inst["durations"]

    # Brute-force baseline
    print("  bruteforce...")
    brute = solve_bruteforce(durations)
    optimal_makespan = brute["optimal_makespan"]
    n_feasible = brute["n_feasible"]
    print(f"  → optimal makespan: {optimal_makespan}, feasible: {n_feasible}")

    # Annealing
    print("  anneal_multi...")
    ann = anneal_multi(
        durations,
        seeds=[0],
        num_reads=200,
        penalty_weight=None,
    )
    best_ann_energy = ann["best_overall_energy"]
    feasibility_rate = ann["feasibility_rate"]
    print(f"  → energy: {best_ann_energy:.3f}, feasibility: {feasibility_rate:.1%}")

    # Exact enumeration (ground truth for small)
    print("  exact_enumeration...")
    exact = exact_enumeration(durations, penalty_weight=None)
    n_total_samples = exact["n_total"]
    n_feasible_ex = sum(1 for r in exact["all_samples"] if r["valid"])
    print(f"  → {n_feasible_ex}/{n_total_samples} feasible")

    # Validation checks
    assert optimal_makespan == 4.0, f"Expected makespan=4.0, got {optimal_makespan}"
    assert n_feasible > 0, "No feasible assignments found"
    assert feasibility_rate >= 0.95, f"Feasibility rate too low: {feasibility_rate:.1%}"

    # Save results
    out = {
        "instance": inst_path.stem,
        "n_jobs": inst["n_jobs"],
        "n_machines": inst["n_machines"],
        "durations": durations.tolist(),
        "optimal_makespan": optimal_makespan,
        "n_feasible_assignments": n_feasible,
        "anneal": {
            "best_energy": best_ann_energy,
            "feasibility_rate": feasibility_rate,
            "num_reads_per_seed": 200,
        },
        "exact_enumeration": {
            "n_total_samples": n_total_samples,
            "n_feasible": n_feasible_ex,
        },
    }

    out_path = results_dir / "integration_small.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  → saved to {out_path}")


if __name__ == "__main__":
    main()

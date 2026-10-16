"""Sweep penalty weights and report feasibility rate at each."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.qubo import build_qubo, decode, energy
from src.anneal import anneal


def load_instance(path: Path) -> dict:
    with open(path) as f:
        data = json.load(f)
    return {
        "name": path.stem,
        "durations": np.array(data["durations"]),
        "n_jobs": data["n_jobs"],
        "n_machines": data["n_machines"],
    }


WEIGHTS = [1.0, 5.0, 10.0, 50.0, 200.0]


def main():
    instance_dir = Path(__file__).parent.parent / "instance"
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    for inst_path in sorted(instance_dir.glob("*.json")):
        print(f"\n=== {inst_path.stem} ===")
        inst = load_instance(inst_path)
        durations = inst["durations"]

        out = {"instance": inst["name"], "weights": []}

        for w in WEIGHTS:
            print(f"  weight={w:.1f}...")
            result = anneal(
                durations, num_reads=500, penalty_weight=w, seed=0
            )
            best = result["best"]
            feasible = sum(1 for s in result["all_samples"] if s["valid"])

            entry = {
                "penalty_weight": w,
                "feasibility_rate": feasible / result["n_reads"],
                "best_energy": best["energy"] if best else None,
                "best_assignment": best["assignment"] if best else None,
            }
            out["weights"].append(entry)

        out_path = results_dir / f"penalty_sweep_{inst['name']}.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  → {out_path}")


if __name__ == "__main__":
    main()

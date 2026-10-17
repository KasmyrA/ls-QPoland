"""Run QAOA depth-sweep on all instances and save results."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.qaoa import qaoa_depth_sweep
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

        # Brute-force baseline
        brute = solve_bruteforce(durations)
        optimal_makespan = brute["optimal_makespan"]

        # QAOA depth sweep (p=1,2,3) — only for small instances
        n_bits = durations.shape[0] * inst["n_machines"]
        if n_bits <= 9:
            print("  qaoa_depth_sweep...")
            qs = qaoa_depth_sweep(durations, penalty_weight=10.0)

            out = {
                "instance": inst["name"],
                "optimal_makespan": optimal_makespan,
                "qaoa_sweep": [],
            }
            for entry in qs["sweep"]:
                out["qaoa_sweep"].append({
                    "depth": entry["depth"],
                    "best_energy": entry["best_energy"],
                    "params": entry["params"],
                    "n_qubits": entry["n_qubits"],
                })

            out_path = results_dir / f"qaoa_{inst['name']}.json"
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)
            print(f"  → {out_path}")
        else:
            print(f"  SKIPPED (n_bits={n_bits} > 9)")


if __name__ == "__main__":
    main()

"""Test script for running VeloxQ solver on Job-Shop QUBO instances."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from load import load_instance
from save import save_schedule
from algorithms.velox_solver import solve_with_veloxq


def main():
    raw_token = os.environ.get("VELOX_TOKEN", "")
    if not raw_token:
        print("\n" + "!" * 60)
        print("[ERROR] VELOX_TOKEN environment variable is not set!")
        print("Please set it in PowerShell before running:")
        print("    $env:VELOX_TOKEN = 'your_token_here'")
        print("!" * 60 + "\n")
        return

    # Clean whitespace, newlines (\r, \n), and enclosing quotes
    clean_token = (
        raw_token.strip()
        .replace("\r", "")
        .replace("\n", "")
        .replace("\t", "")
        .replace("\ufeff", "")
        .replace("\u200b", "")
    )
    if (clean_token.startswith('"') and clean_token.endswith('"')) or (
        clean_token.startswith("'") and clean_token.endswith("'")
    ):
        clean_token = clean_token[1:-1].strip()

    os.environ["VELOX_TOKEN"] = clean_token

    print(">>> Loading instance_2x2.json...")
    inst_path = root_dir / "resources" / "instance_3x3.json"
    instance = load_instance(inst_path)

    print(f">>> Problem: {instance.instance_name} with {instance.n_jobs} jobs on machines {instance.machines}")
    print(">>> Submitting QUBO to VeloxQ Cloud H100 backend...")

    try:
        state = solve_with_veloxq(
            instance=instance,
            solver_type="veloxq",   # or 'sbm'
            num_rep=1000,
            job_name="test_run_3x3",
        )

        print("\n" + "=" * 50)
        print(">>> VELOXQ EXECUTION SUCCESSFUL!")
        print(f"Method: {state.method}")
        print(f"Optimal Makespan: {state.makespan}")
        print(f"Best Energy: {state.metadata.get('energy')}")
        print("Schedule:")
        for task in state.schedule:
            print(f"  Job {task.job} on Machine {task.machine}: [{task.start} -> {task.end}]")
        print("=" * 50)

        # Save output in standard format
        out_file = root_dir / "resources" / "result_veloxq_3x3.json"
        save_schedule(state, out_file)
        print(f"\nSaved result to: {out_file}")

    except Exception as e:
        print(f"\n[ERROR during VeloxQ run]: {e}")


if __name__ == "__main__":
    main()


"""Run every experiment in deterministic order and save outputs."""
import subprocess
from pathlib import Path


def main():
    here = Path(__file__).parent
    root = here.parent

    scripts = [
        "run_annealing.py",
        "run_qaoa.py",
        "run_penalty_sweep.py",
    ]

    for script in scripts:
        print(f"\n{'='*60}")
        print(f">>> Running {script}")
        print(f"{'='*60}")
        result = subprocess.run(
            ["python", str(here / script)],
            cwd=str(root),
        )
        if result.returncode != 0:
            print(f"[ERROR] {script} exited with code {result.returncode}")
            break


if __name__ == "__main__":
    main()

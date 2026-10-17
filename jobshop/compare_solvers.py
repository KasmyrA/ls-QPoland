"""Benchmark and comparison between Classical (CP-SAT) and Quantum (QAOA) solvers."""
from __future__ import annotations

import sys
import time

try:
    from .models import make_test_instance
    from .cp_sat_solver import solve_cp_sat
    from .qubo_builder import build_job_shop_qubo
    from .qaoa_solver import solve_with_qaoa
    from .qubo_decoder import decode_qubo_solution
except ImportError:
    from models import make_test_instance  # type: ignore
    from cp_sat_solver import solve_cp_sat  # type: ignore
    from qubo_builder import build_job_shop_qubo  # type: ignore
    from qaoa_solver import solve_with_qaoa  # type: ignore
    from qubo_decoder import decode_qubo_solution  # type: ignore


def run_comparison(instance):
    results = []

    # -------------------------------------------------
    # 1. CP-SAT
    # -------------------------------------------------
    print("\n=== CP-SAT ===")
    start = time.perf_counter()

    cp_result = solve_cp_sat(
        instance,
        time_limit_seconds=30,
    )

    cp_runtime = time.perf_counter() - start

    print("Status:", cp_result["status"])
    print("Makespan:", cp_result["makespan"])
    print("Runtime:", round(cp_runtime, 3), "seconds")

    results.append({
        "solver": "CP-SAT",
        "status": cp_result["status"],
        "makespan": cp_result["makespan"],
        "runtime": cp_runtime,
    })

    # -------------------------------------------------
    # 2. Build QUBO once
    # -------------------------------------------------
    print("\n=== Building QUBO ===")

    # Adapt this unpacking to your actual function.
    Q, metadata = build_job_shop_qubo(instance)

    # Make sure this is the offset used in QUBO energy.
    offset = metadata.get("offset", 0.0)

    print("Number of binary variables:", Q.shape[0])

    # -------------------------------------------------
    # 3. QAOA
    # -------------------------------------------------
    print("\n=== QAOA ===")
    start = time.perf_counter()

    qaoa_result = solve_with_qaoa(
        Q,
        offset=offset,
        reps=1,
        maxiter=100,
    )

    qaoa_runtime = time.perf_counter() - start

    # Decode into a scheduling solution.
    decoded = decode_qubo_solution(
        qaoa_result["bits"],
        metadata,
    )

    # Adapt these fields to your decoder's return type.
    qaoa_makespan = decoded["makespan"]

    print("QUBO energy:", qaoa_result["energy"])
    print("Makespan:", qaoa_makespan)
    print("Runtime:", round(qaoa_runtime, 3), "seconds")

    results.append({
        "solver": "QAOA",
        "status": qaoa_result["status"],
        "makespan": qaoa_makespan,
        "runtime": qaoa_runtime,
    })

    # -------------------------------------------------
    # 4. Summary
    # -------------------------------------------------
    print("\n=== Comparison ===")
    print(f"{'Solver':<12} {'Status':<12} {'Makespan':>10} {'Time (s)':>12}")

    for result in results:
        print(
            f"{result['solver']:<12} "
            f"{result['status']:<12} "
            f"{str(result['makespan']):>10} "
            f"{result['runtime']:>12.3f}"
        )

    return results


if __name__ == "__main__":
    # Default to 2x2 test instance
    size = sys.argv[1] if len(sys.argv) > 1 else "3x3"
    instance = make_test_instance(size=size)

    run_comparison(instance)


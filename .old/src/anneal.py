"""Simulated-annealing runner using dimod + dwave-neal."""
from __future__ import annotations
import time
import json
import numpy as np
import dimod

from .qubo import build_qubo, decode


def anneal(
    durations: np.ndarray,
    num_reads: int = 1000,
    penalty_weight: float | None = None,
) -> dict:
    """Run simulated annealing and return the best feasible sample."""
    qubo = build_qubo(durations, penalty_weight=penalty_weight)

    bqm = dimod.BinaryQuadraticModel.from_qubo(qubo["qubo"], offset=qubo["offset"])
    sampler = dimod.SimulatedAnnealingSampler()

    sampleset = sampler.sample(
        bqm, num_reads=num_reads,
    )

    # Sort by energy; pick first feasible row-sums-equal-1 sample
    best = None
    all_results: list[dict] = []
    for sample, energy_val in sampleset.data(["sample", "energy"]):
        bits = np.array([sample.get(i, 0) for i in range(qubo["n_bits"])])
        assignment = decode(bits, durations.shape[0])
        valid = all(v >= 0 for v in assignment) and len(set(assignment)) == len(assignment)

        entry = {
            "energy": float(energy_val),
            "bitstring": bits.tolist(),
            "assignment": assignment,
            "valid": valid,
            "infeasible_jobs": sum(1 for v in assignment if v < 0),
        }
        all_results.append(entry)

        if valid and (best is None or energy_val < best["energy"]):
            best = entry

    return {
        "best": best,
        "all_samples": all_results,
        "n_reads": num_reads,
        "penalty_weight": penalty_weight,
        "qubo_offset": qubo["offset"],
        "durations": durations.tolist(),
    }


def anneal_multi(
    durations: np.ndarray,
    seeds=(0, 1, 2),
    num_reads=500,
    penalty_weight=None,
) -> dict:
    """Run annealing with several seeds and return aggregated stats."""
    runs = []
    for s in seeds:
        r = anneal(durations, num_reads=num_reads, penalty_weight=penalty_weight)
        r["seed"] = s
        runs.append(r)

    feasible_energies = [r["best"]["energy"] for r in runs if r["best"]]
    return {
        "runs": runs,
        "best_overall_energy": min(feasible_energies) if feasible_energies else None,
        "feasibility_rate": sum(1 for r in runs if r["best"]) / len(runs),
    }


def exact_enumeration(durations: np.ndarray, penalty_weight=None) -> dict:
    """Brute-force enumerate ALL 2^N samples via dimod exact sampler."""
    qubo = build_qubo(durations, penalty_weight=penalty_weight)
    bqm = dimod.BinaryQuadraticModel.from_qubo(qubo["qubo"], offset=qubo["offset"])

    t0 = time.perf_counter()
    solver = dimod.ExactSolver()
    sampleset = solver.sample(bqm)
    elapsed = time.perf_counter() - t0

    results: list[dict] = []
    for sample, energy_val in sampleset.data(["sample", "energy"]):
        bits = np.array([sample.get(i, 0) for i in range(qubo["n_bits"])])
        assignment = decode(bits, durations.shape[0])
        valid = all(v >= 0 for v in assignment) and len(set(assignment)) == len(assignment)
        results.append({
            "energy": float(energy_val),
            "bitstring": bits.tolist(),
            "assignment": assignment,
            "valid": valid,
        })

    results.sort(key=lambda r: r["energy"])
    best = results[0] if results else None

    return {
        "best": best,
        "all_samples": results,
        "n_total": len(results),
        "elapsed_s": elapsed,
    }

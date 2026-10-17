"""Small p=1 QAOA demonstration for two jobs and two identical machines.

Run from the repository root with:
    python experiments/qaoa_2x2.py
"""
from __future__ import annotations

import itertools

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


N_JOBS = 2
N_MACHINES = 2
DURATIONS = np.array([2.0, 3.0])
PENALTY = 50.0


def build_qubo() -> tuple[np.ndarray, np.ndarray, float]:
    """Return linear terms, pair terms, and offset for the assignment QUBO.

    Variable x[j, m] is stored at index j * N_MACHINES + m.
    The objective is sum of squared machine loads; the penalty enforces one
    machine assignment per job.
    """
    n_bits = N_JOBS * N_MACHINES
    linear = np.zeros(n_bits)
    quadratic = np.zeros((n_bits, n_bits))
    offset = PENALTY * N_JOBS

    for job in range(N_JOBS):
        bits = [job * N_MACHINES + machine for machine in range(N_MACHINES)]
        for bit in bits:
            linear[bit] += DURATIONS[job] ** 2 - PENALTY
        for left, right in itertools.combinations(bits, 2):
            quadratic[left, right] += 2.0 * PENALTY

    for machine in range(N_MACHINES):
        bits = [job * N_MACHINES + machine for job in range(N_JOBS)]
        for left, right in itertools.combinations(bits, 2):
            left_job = left // N_MACHINES
            right_job = right // N_MACHINES
            quadratic[left, right] += 2.0 * DURATIONS[left_job] * DURATIONS[right_job]

    return linear, quadratic, offset


def build_qaoa_circuit(
    gamma: float,
    beta: float,
    linear: np.ndarray,
    quadratic: np.ndarray,
) -> QuantumCircuit:
    """Build a one-layer QAOA circuit with the standard X mixer."""
    circuit = QuantumCircuit(len(linear))
    circuit.h(range(len(linear)))

    # x_i = (1 - Z_i) / 2; the omitted identity terms contribute only
    # a global phase and do not change measurement probabilities.
    for index, coefficient in enumerate(linear):
        circuit.rz(-gamma * coefficient, index)

    for left in range(len(linear)):
        for right in range(left + 1, len(linear)):
            coefficient = quadratic[left, right]
            if coefficient:
                circuit.rz(-gamma * coefficient / 2.0, left)
                circuit.rz(-gamma * coefficient / 2.0, right)
                circuit.rzz(gamma * coefficient / 2.0, left, right)

    circuit.rx(2.0 * beta, range(len(linear)))
    return circuit


def qubo_energy(bits: np.ndarray, linear: np.ndarray, quadratic: np.ndarray, offset: float) -> float:
    value = offset + float(linear @ bits)
    for left in range(len(bits)):
        for right in range(left + 1, len(bits)):
            value += quadratic[left, right] * bits[left] * bits[right]
    return value


def evaluate(
    gamma: float,
    beta: float,
    linear: np.ndarray,
    quadratic: np.ndarray,
    offset: float,
) -> tuple[float, np.ndarray]:
    circuit = build_qaoa_circuit(gamma, beta, linear, quadratic)
    probabilities = np.abs(Statevector.from_instruction(circuit).data) ** 2
    expected_energy = 0.0
    for state, probability in enumerate(probabilities):
        bits = np.array([(state >> index) & 1 for index in range(len(linear))])
        expected_energy += probability * qubo_energy(bits, linear, quadratic, offset)
    return float(expected_energy), probabilities


def main() -> None:
    linear, quadratic, offset = build_qubo()
    best: tuple[float, float, float] | None = None

    # A compact parameter grid keeps this example simple and reproducible.
    for gamma in np.linspace(0.0, 2.0 * np.pi, 41):
        for beta in np.linspace(0.0, np.pi, 41):
            expected_energy, _ = evaluate(gamma, beta, linear, quadratic, offset)
            if best is None or expected_energy < best[0]:
                best = (expected_energy, float(gamma), float(beta))

    if best is None:
        raise RuntimeError("QAOA parameter search produced no result")

    expected_energy, gamma, beta = best
    _, probabilities = evaluate(gamma, beta, linear, quadratic, offset)
    feasible: list[tuple[float, list[int], float]] = []

    for state, probability in enumerate(probabilities):
        bits = np.array([(state >> index) & 1 for index in range(len(linear))])
        assignment = []
        valid = True
        for job in range(N_JOBS):
            row = bits[job * N_MACHINES : (job + 1) * N_MACHINES]
            if row.sum() != 1:
                valid = False
                break
            assignment.append(int(np.argmax(row)))
        if valid:
            loads = [0.0] * N_MACHINES
            for job, machine in enumerate(assignment):
                loads[machine] += DURATIONS[job]
            feasible.append((float(probability), assignment, max(loads)))

    feasible.sort(reverse=True)
    print(f"2 jobs / 2 machines; durations={DURATIONS.tolist()}; QAOA depth p=1")
    print(f"Best expected QUBO energy: {expected_energy:.4f}")
    print(f"Parameters: gamma={gamma:.4f}, beta={beta:.4f}")
    print("Most probable feasible assignments (machine per job):")
    for probability, assignment, makespan in feasible[:4]:
        print(
            f"  P={probability:.4f}, assignment={assignment}, "
            f"makespan={makespan:g}"
        )


if __name__ == "__main__":
    main()

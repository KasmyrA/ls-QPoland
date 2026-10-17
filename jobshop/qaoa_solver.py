"""QAOA (Quantum Approximate Optimization Algorithm) solver for Job-Shop scheduling."""
from __future__ import annotations

from typing import Any, Optional
import numpy as np
from scipy.optimize import minimize

try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False


def qubo_to_ising(Q: np.ndarray, offset: float = 0.0) -> tuple[np.ndarray, np.ndarray, float]:
    """Convert QUBO matrix Q to Ising Hamiltonian parameters.

    QUBO: H(x) = x^T Q x + offset, where x_i in {0, 1}.
    Substitution: x_i = (1 - Z_i) / 2, where Z_i in {+1, -1}.

    Returns
    -------
    h : np.ndarray (N,)
        Linear coefficients for Z_i.
    J : np.ndarray (N, N)
        Pairwise coupling coefficients for Z_i Z_j (i < j).
    C : float
        Scalar energy offset.
    """
    N = Q.shape[0]
    # Symmetrize Q
    W = 0.5 * (Q + Q.T)

    h = np.zeros(N, dtype=float)
    J = np.zeros((N, N), dtype=float)
    C = float(offset)

    # Diagonal contributions: W_ii * (1 - Z_i)/2 = W_ii/2 - (W_ii/2)*Z_i
    diag = np.diag(W)
    C += 0.5 * float(np.sum(diag))
    h -= 0.5 * diag

    # Off-diagonal contributions:
    # 2 * W_ij * (1 - Z_i)(1 - Z_j)/4 = (W_ij/2) * (1 - Z_i - Z_j + Z_i*Z_j)
    for i in range(N):
        for j in range(i + 1, N):
            w_ij = W[i, j]
            if abs(w_ij) < 1e-12:
                continue
            C += 0.5 * w_ij
            h[i] -= 0.5 * w_ij
            h[j] -= 0.5 * w_ij
            J[i, j] = 0.5 * w_ij
            J[j, i] = 0.5 * w_ij

    return h, J, C


def build_qaoa_circuit(
    h: np.ndarray,
    J: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
) -> QuantumCircuit:
    """Build a QAOA quantum circuit of depth p using explicit gates.

    Parameters
    ----------
    h : (N,) linear Z coefficients
    J : (N, N) pairwise ZZ coefficients
    gamma : (p,) cost unitary rotation angles
    beta : (p,) mixer unitary rotation angles
    """
    n_qubits = len(h)
    p = len(gamma)
    qc = QuantumCircuit(n_qubits)

    # Initial uniform superposition |+>^n
    qc.h(range(n_qubits))

    for l in range(p):
        g = float(gamma[l])
        b = float(beta[l])

        # 1. Cost unitary: exp(-i * gamma * H_C)
        # Single-qubit Z rotations: exp(-i * g * h_i * Z_i) = RZ(2 * g * h_i)
        for i in range(n_qubits):
            if abs(h[i]) > 1e-12:
                qc.rz(2.0 * g * h[i], i)

        # Two-qubit ZZ interactions: exp(-i * g * J_ij * Z_i Z_j) = RZZ(2 * g * J_ij)
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                if abs(J[i, j]) > 1e-12:
                    # CX - RZ - CX implementation of RZZ
                    qc.cx(i, j)
                    qc.rz(2.0 * g * J[i, j], j)
                    qc.cx(i, j)

        # 2. Mixer unitary: exp(-i * beta * sum(X_i)) = RX(2 * beta)
        for i in range(n_qubits):
            qc.rx(2.0 * b, i)

    return qc


def solve_with_qaoa(
    Q: np.ndarray,
    offset: float = 0.0,
    reps: int = 1,
    maxiter: int = 100,
) -> dict[str, Any]:
    """Solve the Job-Shop QUBO using QAOA on Qiskit's statevector simulator.

    Parameters
    ----------
    Q : np.ndarray
        QUBO matrix of shape (n_vars, n_vars).
    offset : float
        QUBO energy offset.
    reps : int
        Number of QAOA layers (p).
    maxiter : int
        Maximum iterations for classical optimizer (COBYLA).

    Returns
    -------
    dict
        Contains:
        - 'bits': optimal bitstring (np.ndarray of ints)
        - 'energy': QUBO objective value at 'bits'
        - 'status': optimizer status string
        - 'gamma': optimized cost parameters
        - 'beta': optimized mixer parameters
    """
    if not HAS_QISKIT:
        raise ImportError("qiskit is required to run solve_with_qaoa")

    N = Q.shape[0]
    h, J, const = qubo_to_ising(Q, offset)

    # Precompute QUBO energy for all 2^N basis states for fast evaluation
    # Qiskit qubit convention: qubit i is (k >> i) & 1
    basis_states = 1 << N
    all_bits = np.zeros((basis_states, N), dtype=float)
    for k in range(basis_states):
        for i in range(N):
            all_bits[k, i] = (k >> i) & 1

    # Exact QUBO energy for each basis state: E[k] = x_k^T Q x_k + offset
    # Vectorized computation: diag(all_bits @ Q @ all_bits.T) + offset
    exact_energies = np.sum((all_bits @ Q) * all_bits, axis=1) + offset

    # Objective function: expectation value <H_C> = sum_k P(k) * E(k)
    def cost_function(params: np.ndarray) -> float:
        gamma = params[:reps]
        beta = params[reps:]
        qc = build_qaoa_circuit(h, J, gamma, beta)
        sv = Statevector.from_int(0, basis_states).evolve(qc)
        probs = np.abs(sv.data) ** 2
        return float(np.dot(probs, exact_energies))

    # Initial parameter guess
    init_params = np.concatenate([
        np.full(reps, 0.5),   # initial gamma
        np.full(reps, 0.5),   # initial beta
    ])

    # Optimize using COBYLA
    opt_result = minimize(
        cost_function,
        init_params,
        method="COBYLA",
        options={"maxiter": maxiter},
    )

    opt_gamma = opt_result.x[:reps]
    opt_beta = opt_result.x[reps:]

    # Sample best bitstring from optimized circuit
    opt_qc = build_qaoa_circuit(h, J, opt_gamma, opt_beta)
    final_sv = Statevector.from_int(0, basis_states).evolve(opt_qc)
    final_probs = np.abs(final_sv.data) ** 2

    # Pick the state with minimum energy among candidate states
    # Candidates: states with probability >= 10% of maximum probability
    prob_threshold = 0.1 * np.max(final_probs)
    candidate_indices = np.where(final_probs >= prob_threshold)[0]
    if len(candidate_indices) == 0:
        best_k = int(np.argmax(final_probs))
    else:
        best_k = int(candidate_indices[np.argmin(exact_energies[candidate_indices])])

    best_bits = all_bits[best_k].astype(int)
    best_energy = float(exact_energies[best_k])

    return {
        "bits": best_bits,
        "energy": best_energy,
        "status": "OPTIMAL" if opt_result.success else "FEASIBLE",
        "gamma": opt_gamma.tolist(),
        "beta": opt_beta.tolist(),
        "expected_energy": float(opt_result.fun),
        "n_qubits": N,
    }


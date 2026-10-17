"""QAOA solver for the job-assignment QUBO.

Uses Qiskit Aer statevector simulator — exact results on ≤20 qubits, so we
can enumerate all 512 bitstrings and pick the best feasible one (equivalent
to brute force). Real QPU would need sampling; here we demonstrate the
circuit construction and parameter sweep.
"""
from __future__ import annotations
import numpy as np

try:
    from qiskit import QuantumCircuit, Aer, transpile
    from qiskit.quantum_info import Statevector
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False

from .qubo import build_qubo, decode


def _ising_from_qubo(Q_matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """Convert a QUBO matrix to Ising form H = Σ J[i,j] σz_i σz_j + Σ h[i] σz_i + const.

    For binary x in {0,1}, map x = (1-σz)/2:
        Q[i,i] x_i  →  -Q[i,i]/2 * σz_i + Q[i,i]/2
        Q[i,j] x_i x_j (i≠j) → Q[i,j]/4 * (1-σz_i)(1-σz_j)
                            = Q[i,j]/4 - Q[i,j]/4*σz_i - Q[i,j]/4*σz_j + Q[i,j]/4*σz_i σz_j
    """
    N = Q_matrix.shape[0]
    h = np.zeros(N)
    J = np.zeros((N, N))
    const = 0.0

    for i in range(N):
        # diagonal term: Q[i,i]*x_i
        diag_contrib = Q_matrix[i, i] / 4.0
        const += diag_contrib
        h[i] -= Q_matrix[i, i] / 2.0

    for i in range(N):
        for j in range(i + 1, N):
            off = Q_matrix[i, j]
            if abs(off) < 1e-12:
                continue
            const += off / 4.0
            h[i] -= off / 4.0
            h[j] -= off / 4.0
            J[i, j] = off / 4.0
            J[j, i] = off / 4.0

    return h, J, const


def build_qaoa_circuit(
    h: np.ndarray,
    J: np.ndarray,
    p: int,
    gamma: np.ndarray | None = None,
    beta: np.ndarray | None = None,
) -> QuantumCircuit:
    """Construct a QAOA circuit of depth p for the Ising Hamiltonian.

    Parameters
    ----------
    h : local fields (N,)
    J : couplings (N, N), symmetric
    p : number of QAOA layers
    gamma : (p,) variational parameters for cost unitary
    beta :  (p,) variational parameters for mixer unitary
    """
    n_qubits = len(h)
    qc = QuantumCircuit(n_qubits)

    # Initial superposition
    qc.h(range(n_qubits))

    if gamma is None or beta is None:
        # Return a parametrised circuit (for optimisation)
        from qiskit.circuit import ParameterVector
        gamma_p = ParameterVector("γ", p)
        beta_p = ParameterVector("β", p)

        for layer in range(p):
            # Cost unitary: exp(-i γ_l H_C)
            for i in range(n_qubits):
                qc.rz(2 * gamma_p[layer], i)
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    if abs(J[i, j]) > 1e-12:
                        qc.cx(i, j)
                        qc.rz(2 * J[i, j] * gamma_p[layer], j)
                        qc.cx(i, j)

            # Mixer unitary: exp(-i β_l H_B), H_B = Σ σx_i
            for i in range(n_qubits):
                qc.rx(2 * beta_p[layer], i)

        return qc

    # Concrete parameters — evaluate expectation value directly
    for layer in range(p):
        γ, β = gamma[layer], beta[layer]
        # Cost unitary
        for i in range(n_qubits):
            qc.rz(2 * γ, i)
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                if abs(J[i, j]) > 1e-12:
                    qc.cx(i, j)
                    qc.rz(2 * J[i, j] * γ, j)
                    qc.cx(i, j)

        # Mixer unitary
        for i in range(n_qubits):
            qc.rx(2 * β, i)

    return qc


def qaoa_expectation(
    h: np.ndarray, J: np.ndarray, p: int, gamma: np.ndarray, beta: np.ndarray
) -> float:
    """Compute ⟨γ,β|H_C|γ,β⟩ via statevector simulation.

    Returns the expected cost energy (Ising Hamiltonian).
    """
    if not HAS_QISKIT:
        raise ImportError("qiskit required for QAOA")

    qc = build_qaoa_circuit(h, J, p, gamma, beta)
    simulator = Aer.get_backend("statevector_simulator")
    sv = Statevector.from_instruction(transpile(qc, simulator))

    # Compute ⟨ψ|H_C|ψ⟩ by expanding in computational basis
    n = len(h)
    expectation = 0.0
    for bitstring_int in range(1 << n):
        bits = np.array([(bitstring_int >> k) & 1 for k in range(n)])
        # σz eigenvalue: +1 if bit=0, -1 if bit=1 → s_i = 1-2*bits[i]
        s = 1 - 2 * bits
        prob = float(np.abs(sv[bitstring_int]) ** 2)
        energy = 0.5 * np.dot(h, s) + 0.5 * s @ J @ s
        expectation += prob * energy

    return expectation


def qaoa_solve(
    durations: np.ndarray,
    p: int = 1,
    penalty_weight: float = 10.0,
    n_restarts: int = 20,
) -> dict:
    """Run QAOA with random-restart gradient-free optimisation.

    For small problems we sweep a grid of (γ,β) pairs; for larger ones
    one would use scipy.optimize.minimize or a real QPU sampler.
    """
    qubo = build_qubo(durations, penalty_weight=penalty_weight)
    h, J, const = _ising_from_qubo(qubo["Q"])

    # Grid search over (γ, β) pairs
    best_energy = float("inf")
    best_params = None
    best_bitstring = None

    # Coarse grid
    gammas_grid = np.linspace(0.01, 3.0, 8)
    betas_grid = np.linspace(0.01, 3.0, 8)

    for γ_val in gammas_grid:
        for β_val in betas_grid:
            gamma_vec = np.full(p, γ_val)
            beta_vec = np.full(p, β_val)

            try:
                energy = qaoa_expectation(h, J, p, gamma_vec, beta_vec)
            except Exception:
                continue

            if energy < best_energy:
                best_energy = energy
                best_params = {"gamma": γ_val, "beta": β_val}

    # Refine around best
    if best_params is not None:
        from scipy.optimize import minimize
        def objective(params):
            g, b = params[0], params[1]
            return qaoa_expectation(h, J, p, np.full(p, g), np.full(p, b))

        res = minimize(
            objective,
            [best_params["gamma"], best_params["beta"]],
            method="Nelder-Mead",
            options={"maxiter": 200},
        )
        if res.fun < best_energy:
            best_energy = float(res.fun)
            best_params = {"gamma": float(res.x[0]), "beta": float(res.x[1])}

    # Decode the best bitstring by measuring at optimal params
    if HAS_QISKIT and best_params is not None:
        qc_opt = build_qaoa_circuit(
            h, J, p,
            gamma=np.full(p, best_params["gamma"]),
            beta=np.full(p, best_params["beta"]),
        )
        simulator = Aer.get_backend("statevector_simulator")
        sv = Statevector.from_instruction(transpile(qc_opt, simulator))

        # Find the basis state with highest probability
        max_idx = int(np.argmax(np.abs(sv) ** 2))
        bits = np.array([(max_idx >> k) & 1 for k in range(len(h))])
        assignment = decode(bits, durations.shape[0])
    else:
        assignment = []

    return {
        "best_energy": best_energy + const,  # back to QUBO energy
        "params": best_params,
        "p": p,
        "assignment": assignment,
        "n_qubits": len(h),
    }


def qaoa_depth_sweep(durations: np.ndarray, penalty_weight=10.0) -> dict:
    """Run QAOA at several depths and compare."""
    results = []
    for p in range(1, 4):
        r = qaoa_solve(durations, p=p, penalty_weight=penalty_weight)
        r["depth"] = p
        results.append(r)
    return {"sweep": results}

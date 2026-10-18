"""Gate-based QAOA sampler (Qiskit statevector simulator) for the QUBO from qubo.py.

Plugs into the same adapter as the other backends:
    solver = make_qubo_solver(make_qaoa_sampler(reps=2, maxiter=150))
Works on any QUBO, but statevector cost is 2^n: use it for small problems (2x2 basic, <= ~16 qubits).

Math (x_i = (1 - Z_i)/2):
    H_C = sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j + C
    U_C(g) = exp(-i g H_C)  -> RZ(2 g h_i) and RZZ(2 g J_ij) = CX-RZ-CX
    U_B(b) = exp(-i b sum X_i) -> RX(2 b)
    |psi(g,b)> = prod_l U_B(b_l) U_C(g_l) |+>^n ;  cost = <psi|E|psi>, E(k) = x_k^T Q x_k + offset
Qiskit ordering: qubit i <-> bit i of the basis index k.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    HAS_QISKIT = True
except ImportError:  # pragma: no cover
    HAS_QISKIT = False


def qubo_to_ising(Q: np.ndarray, offset: float = 0.0):
    """Q -> (h, J, C) with  x^T Q x + offset = sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j + C."""
    N = Q.shape[0]
    W = 0.5 * (Q + Q.T)
    h = np.zeros(N)
    J = np.zeros((N, N))
    C = float(offset)
    diag = np.diag(W)
    C += 0.5 * float(diag.sum())
    h -= 0.5 * diag
    for i in range(N):
        for j in range(i + 1, N):
            w = W[i, j]
            if abs(w) < 1e-12:
                continue
            C += 0.5 * w
            h[i] -= 0.5 * w
            h[j] -= 0.5 * w
            J[i, j] = J[j, i] = 0.5 * w
    return h, J, C


def build_qaoa_circuit(h, J, gamma, beta) -> "QuantumCircuit":
    n = len(h)
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for g, b in zip(gamma, beta):
        for i in range(n):
            if abs(h[i]) > 1e-12:
                qc.rz(2.0 * g * h[i], i)
        for i in range(n):
            for j in range(i + 1, n):
                if abs(J[i, j]) > 1e-12:
                    qc.cx(i, j)
                    qc.rz(2.0 * g * J[i, j], j)
                    qc.cx(i, j)
        for i in range(n):
            qc.rx(2.0 * b, i)
    return qc


def make_qaoa_sampler(reps: int = 2, maxiter: int = 150, top_k: int = 256,
                      max_qubits: int = 16, init_gamma: float = 0.5, init_beta: float = 0.5):
    """Returns sampler(Q, offset) -> [(bits, energy)] using the top_k most probable basis states
    after optimising the 2p QAOA angles with COBYLA on the expectation <E>."""
    if not HAS_QISKIT:
        raise ImportError("qiskit is required: pip install qiskit")

    def sample(Q: np.ndarray, offset: float = 0.0):
        n = Q.shape[0]
        if n > max_qubits:
            raise ValueError(f"QAOA statevector limited to {max_qubits} qubits, problem has {n}")
        Qs = 0.5 * (Q + Q.T)
        h, J, _ = qubo_to_ising(Qs, offset)
        dim = 1 << n
        k_idx = np.arange(dim)
        bits = ((k_idx[:, None] >> np.arange(n)) & 1).astype(int)
        E = np.einsum("ki,ij,kj->k", bits.astype(float), Qs, bits.astype(float)) + offset

        def probs(params):
            qc = build_qaoa_circuit(h, J, params[:reps], params[reps:])
            return np.abs(Statevector(qc).data) ** 2       # Statevector(qc) starts from |0...0>

        res = minimize(lambda p: float(probs(p) @ E),
                       np.concatenate([np.full(reps, init_gamma), np.full(reps, init_beta)]),
                       method="COBYLA", options={"maxiter": maxiter})
        p = probs(res.x)
        order = np.argsort(-p)[:top_k]
        out = [(bits[k].copy(), float(E[k])) for k in order if p[k] > 1e-9]
        out.sort(key=lambda t: t[1])
        sample.last_info = {"n_qubits": n, "reps": reps, "expected_energy": float(res.fun),
                            "optimizer_success": bool(res.success), "nfev": int(res.nfev)}
        return out

    sample.wants_md = False
    sample.max_vars = max_qubits
    sample.last_info = {}
    return sample
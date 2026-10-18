"""Local QUBO samplers (no cloud needed): exact brute force and vectorised simulated annealing.

A sampler is a callable  sampler(Q, offset) -> list[(bits: np.ndarray[int], energy: float)]
(the VeloxQ sampler in veloxq_solver.py has the same interface).
"""
from __future__ import annotations

import numpy as np


def _energy(Q: np.ndarray, offset: float, X: np.ndarray) -> np.ndarray:
    return np.einsum("ri,ij,rj->r", X, Q, X) + offset


def make_exact_sampler(max_vars: int = 24):
    """Brute force over all 2^n assignments - true QUBO ground state (tiny problems only)."""
    def sample(Q, offset=0.0):
        n = Q.shape[0]
        if n > max_vars:
            raise ValueError(f"exact sampler limited to {max_vars} variables, got {n}")
        best_e, best_x = np.inf, None
        chunk = 1 << min(n, 16)
        for start in range(0, 1 << n, chunk):
            ids = np.arange(start, min(start + chunk, 1 << n))
            X = ((ids[:, None] >> np.arange(n)) & 1).astype(float)
            e = _energy(Q, offset, X)
            j = int(np.argmin(e))
            if e[j] < best_e:
                best_e, best_x = float(e[j]), X[j].astype(int)
        return [(best_x, best_e)]
    sample.max_vars = max_vars
    return sample


def _fill_slack(coeffs: list[int], rest: int) -> list[int]:
    """Bits of a slack encoding whose weighted sum equals `rest` (0 <= rest <= sum(coeffs))."""
    for mask in range(1 << len(coeffs)):
        if sum(c for b, c in enumerate(coeffs) if mask >> b & 1) == rest:
            return [(mask >> b) & 1 for b in range(len(coeffs))]
    return [0] * len(coeffs)


def make_sa_sampler(num_reads: int = 64, sweeps: int = 500, seed: int = 0,
                    t_hi: float | None = None, t_lo: float = 0.3, init_density: float = 0.05,
                    structured: bool = True):
    """Vectorised simulated annealing (all reads annealed in parallel).

    structured=True: when the builder metadata `md` is passed, the duty-cycle slack bits are NOT annealed
    (single bit flips cannot move a slack together with the work it balances). They are eliminated
    analytically - for a given schedule the optimal slack is max(0, X-W), so the penalty is
    lambda*max(0, W-X)^2 - and filled in exactly at the end, so returned samples are ordinary full QUBO
    assignments with the true QUBO energy. Cloud solvers (VeloxQ) receive the plain slack QUBO instead.
    """
    def sample(Q, offset=0.0, md=None):
        if structured and md is not None and md.get("duty_windows"):
            return _sample_structured(Q, offset, md)
        return _sample_plain(Q, offset)

    def _sample_structured(Q, offset, md):
        n_x, lam = md["n_x"], md["penalty_lambda"]
        Qc = 0.5 * (md["Q_core"] + md["Q_core"].T)
        d = np.diag(Qc).copy()
        W = Qc.copy(); np.fill_diagonal(W, 0.0)
        wins = md["duty_windows"]
        Xw = np.array([w["X"] for w in wins], dtype=float)
        inc_w = [[] for _ in range(n_x)]; inc_c = [[] for _ in range(n_x)]
        for wi, w in enumerate(wins):
            for i, c in w["coeffs"]:
                inc_w[i].append(wi); inc_c[i].append(c)
        inc_w = [np.array(a, dtype=int) for a in inc_w]; inc_c = [np.array(a, dtype=float) for a in inc_c]
        rng = np.random.default_rng(seed)
        R = num_reads
        x = (rng.random((R, n_x)) < init_density).astype(float)
        F = x @ W
        Wk = np.zeros((R, len(wins)))
        for wi, w in enumerate(wins):
            for i, c in w["coeffs"]:
                Wk[:, wi] += c * x[:, i]
        hi = t_hi if t_hi is not None else max(2.0 * lam, 1.0)
        temps = np.geomspace(hi, t_lo, sweeps)
        for T in np.concatenate([temps, np.zeros(3)]):
            for i in rng.permutation(n_x):
                dx = 1.0 - 2.0 * x[:, i]
                delta = dx * (d[i] + 2.0 * F[:, i])
                wi_, ci_ = inc_w[i], inc_c[i]
                if len(wi_):
                    old = np.maximum(0.0, Wk[:, wi_] - Xw[wi_]) ** 2
                    new = np.maximum(0.0, Wk[:, wi_] + dx[:, None] * ci_ - Xw[wi_]) ** 2
                    delta = delta + lam * (new - old).sum(axis=1)
                if T > 0:
                    acc = (delta <= 0) | (rng.random(R) < np.exp(-np.clip(delta, 0, None) / T))
                else:
                    acc = delta < 0
                if acc.any():
                    x[acc, i] += dx[acc]
                    F[acc] += dx[acc, None] * W[i][None, :]
                    if len(wi_):
                        Wk[np.ix_(acc, wi_)] += dx[acc, None] * ci_
        out = []
        for r in range(R):
            bits = np.zeros(Q.shape[0], dtype=int)
            bits[:n_x] = x[r].astype(int)
            for wi, w in enumerate(wins):
                rest = int(max(0.0, Xw[wi] - Wk[r, wi]))
                coeffs = [a for _, a in w["slack"]]
                for (si, _), bv in zip(w["slack"], _fill_slack(coeffs, rest)):
                    bits[si] = bv
            out.append((bits, float(bits @ Q @ bits + offset)))
        out.sort(key=lambda t: t[1])
        return out

    def _sample_plain(Q, offset=0.0):
        n = Q.shape[0]
        rng = np.random.default_rng(seed)
        Qs = 0.5 * (Q + Q.T)
        d = np.diag(Qs).copy()
        W = Qs.copy()
        np.fill_diagonal(W, 0.0)
        R = num_reads
        x = (rng.random((R, n)) < init_density).astype(float)
        F = x @ W                                  # field, W symmetric
        hi = t_hi if t_hi is not None else 0.5 * float(np.abs(Qs).max())
        temps = np.geomspace(max(hi, t_lo * 2), t_lo, sweeps)
        for T in np.concatenate([temps, np.zeros(3)]):   # 3 greedy sweeps at the end
            for i in rng.permutation(n):
                dx = 1.0 - 2.0 * x[:, i]
                delta = dx * (d[i] + 2.0 * F[:, i])
                if T > 0:
                    acc = (delta <= 0) | (rng.random(R) < np.exp(-np.clip(delta, 0, None) / T))
                else:
                    acc = delta < 0
                if acc.any():
                    x[acc, i] += dx[acc]
                    F[acc] += dx[acc, None] * W[i][None, :]
        e = _energy(Qs, offset, x)
        order = np.argsort(e)
        return [(x[r].astype(int), float(e[r])) for r in order]
    sample.wants_md = True
    return sample
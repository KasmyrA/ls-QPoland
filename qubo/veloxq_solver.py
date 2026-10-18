"""VeloxQ Cloud sampler (QUBO -> dimod BQM -> veloxq_sdk) with the same interface as samplers.py.

    sampler = make_veloxq_sampler(solver_type="sbm", num_rep=2000)        # or "veloxq"
    solver  = make_qubo_solver(sampler)                                   # plugs into every pipeline

Authentication: VELOX_TOKEN environment variable or token="...".
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

try:
    import dimod
except ImportError:  # pragma: no cover
    dimod = None

try:
    from veloxq_sdk import VeloxQH100_1, VeloxQParameters, VeloxQSolver, SBMParameters, SBMSolver
    from veloxq_sdk.config import VeloxQAPIConfig
    HAS_VELOXQ = True
except ImportError:  # pragma: no cover
    HAS_VELOXQ = False


def configure_token(token: Optional[str] = None) -> str: #no explanation needed
    """Read + sanitise the API token (strip whitespace, quotes, zero-width chars) and register it."""
    raw = token or os.environ.get("VELOX_TOKEN", "")
    if not raw:
        raise ValueError("VELOX_TOKEN not found: set the environment variable or pass token='...'")
    tok = (raw.strip().replace("\r", "").replace("\n", "").replace("\t", "")
           .replace("\ufeff", "").replace("\u200b", ""))
    if len(tok) >= 2 and tok[0] == tok[-1] and tok[0] in "\"'":
        tok = tok[1:-1].strip()
    os.environ["VELOX_TOKEN"] = tok
    VeloxQAPIConfig.instance().token = tok
    return tok


def qubo_to_bqm(Q: np.ndarray, offset: float = 0.0): #Qubo problem = sum(q[i,i]*x[i] + sum(q[i,j]*x[i]*x[j]) linear and quadratic terms
    """Symmetric QUBO matrix -> dimod.BinaryQuadraticModel (every variable is declared)."""
    if dimod is None:
        raise ImportError("dimod is not installed: pip install dimod")
    n = Q.shape[0]
    linear = {i: float(Q[i, i]) for i in range(n)}
    quadratic = {}
    for i in range(n):
        row = Q[i, i + 1:] + Q[i + 1:, i]
        for j in np.nonzero(np.abs(row) > 1e-12)[0]:
            quadratic[(i, i + 1 + int(j))] = float(row[j])
    return dimod.BinaryQuadraticModel(linear, quadratic, float(offset), vartype=dimod.BINARY)


def make_veloxq_sampler(solver_type: str = "sbm", num_rep: int = 1000, timeout: int = 60,
                        token: Optional[str] = None, runs: int = 1, name: str = "jobshop"):
    """solver_type: 'veloxq' (VeloxQSolver) or 'sbm' (SBMSolver, simulated bifurcation).

    Every call submits the BQM `runs` times and returns ALL returned samples; the caller filters
    for feasibility, which matters because the lowest-energy sample is not always feasible.
    """
    if not HAS_VELOXQ:
        raise ImportError("veloxq_sdk is not installed: pip install git+https://github.com/quantumz-io/veloxq_sdk.git")
    configure_token(token)

    def sample(Q: np.ndarray, offset: float = 0.0):
        n = Q.shape[0]
        bqm = qubo_to_bqm(Q, offset)
        backend = VeloxQH100_1()
        if solver_type.lower() == "sbm":
            solver = SBMSolver(parameters=SBMParameters(num_rep=num_rep), backend=backend)
        else:
            solver = VeloxQSolver(parameters=VeloxQParameters(num_rep=num_rep, timeout=timeout), backend=backend)
        out = []
        for r in range(runs):
            ss = solver.sample(bqm, name=f"{name}_{r}", vartype="BINARY")
            try:
                records = [(dict(rec.sample), float(rec.energy)) for rec in ss.data(["sample", "energy"])]
            except AttributeError:                      # SDK object without .data(): keep the best only
                records = [(dict(ss.first.sample), float(ss.first.energy))]
            for smp, energy in records:
                x = np.zeros(n, dtype=int)
                for v, val in smp.items():
                    if int(v) < n:
                        x[int(v)] = int(val)
                out.append((x, energy))
        return out
    return sample
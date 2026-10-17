"""VeloxQ Cloud Solver integration for Job-Shop / ERP QUBO problems.

Uses `veloxq_sdk` to submit QUBO problems to the VeloxQ quantum/classical-accelerated
H100 backends via cloud API.

Authentication:
    Automatically uses `os.environ.get("VELOX_TOKEN")` or an explicitly provided token.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Union
import numpy as np

import dimod

try:
    from veloxq_sdk import (
        VeloxQH100_1,
        VeloxQParameters,
        VeloxQSolver,
        SBMParameters,
        SBMSolver,
    )
    from veloxq_sdk.config import VeloxQAPIConfig
    HAS_VELOXQ = True
except ImportError:
    HAS_VELOXQ = False

try:
    from ..entities import CalculationState, InstanceData
    from .qubo import build_job_shop_qubo, decode_qubo_solution
except (ImportError, ValueError):
    try:
        from controlled_code.entities import CalculationState, InstanceData  # type: ignore
        from controlled_code.algorithms.qubo import build_job_shop_qubo, decode_qubo_solution  # type: ignore
    except ImportError:
        from entities import CalculationState, InstanceData  # type: ignore
        from algorithms.qubo import build_job_shop_qubo, decode_qubo_solution  # type: ignore


def qubo_to_bqm(qubo_info: dict[str, Any]) -> dimod.BinaryQuadraticModel:
    """Convert qubo_info from build_job_shop_qubo to a dimod.BinaryQuadraticModel.

    Parameters
    ----------
    qubo_info : dict
        Output dictionary from build_job_shop_qubo containing 'Q' matrix and 'offset'.

    Returns
    -------
    dimod.BinaryQuadraticModel
        Binary quadratic model ready for dimod samplers and VeloxQSolver.
    """
    Q = qubo_info["Q"]
    offset = float(qubo_info["offset"])
    n = Q.shape[0]

    linear: dict[int, float] = {}
    quadratic: dict[tuple[int, int], float] = {}

    for i in range(n):
        diag_val = float(Q[i, i])
        if abs(diag_val) > 1e-12:
            linear[i] = diag_val

        for j in range(i + 1, n):
            val = float(Q[i, j] + Q[j, i])
            if abs(val) > 1e-12:
                quadratic[(i, j)] = val

    return dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype=dimod.BINARY)


def solve_with_veloxq(
    instance: InstanceData,
    horizon: Optional[int] = None,
    penalty_lambda: Optional[float] = None,
    token: Optional[str] = None,
    solver_type: str = "veloxq",  # 'veloxq' or 'sbm'
    num_rep: int = 1000,
    timeout: int = 60,
    job_name: Optional[str] = None,
) -> CalculationState:
    """Solve the Job-Shop QUBO using VeloxQ Cloud API on NVIDIA H100 backends.

    Parameters
    ----------
    instance : InstanceData
        Problem instance (e.g. from load_instance).
    horizon : int, optional
        Scheduling time horizon H. If None, auto-calculated.
    penalty_lambda : float, optional
        Constraint penalty multiplier lambda. If None, auto-calculated.
    token : str, optional
        VeloxQ API token. If None, reads from os.environ["VELOX_TOKEN"].
    solver_type : str
        'veloxq' for VeloxQSolver, or 'sbm' for SBMSolver (Simulated Bifurcation Machine).
    num_rep : int
        Number of repetitions/samples to execute (default: 1000).
    timeout : int
        Timeout in seconds for job completion.
    job_name : str, optional
        Descriptive label for the cloud job.

    Returns
    -------
    CalculationState
        Schedule state containing decoded tasks, makespan, and execution metadata.
    """
    if not HAS_VELOXQ:
        raise ImportError(
            "veloxq_sdk is not installed. Please run: pip install veloxq_sdk"
        )

    # 1. Check & configure API token
    raw_token = token or os.environ.get("VELOX_TOKEN", "")
    if not raw_token:
        raise ValueError(
            "VELOX_TOKEN not found! Please set the VELOX_TOKEN environment variable "
            "or pass token='...' to solve_with_veloxq."
        )

    # Sanitize token: strip whitespace, newlines (\r, \n), and enclosing quotes
    api_token = (
        raw_token.strip()
        .replace("\r", "")
        .replace("\n", "")
        .replace("\t", "")
        .replace("\ufeff", "")
        .replace("\u200b", "")
    )
    if (api_token.startswith('"') and api_token.endswith('"')) or (
        api_token.startswith("'") and api_token.endswith("'")
    ):
        api_token = api_token[1:-1].strip()

    # Update environment and SDK config with clean ASCII token
    os.environ["VELOX_TOKEN"] = api_token
    config = VeloxQAPIConfig.instance()
    config.token = api_token

    # 2. Build Job-Shop QUBO
    qubo_info = build_job_shop_qubo(
        instance=instance,
        horizon=horizon,
        penalty_lambda=penalty_lambda,
    )
    bqm = qubo_to_bqm(qubo_info)

    # 3. Initialize Solver and backend
    backend = VeloxQH100_1()
    name = job_name or f"jobshop_{instance.instance_name}"

    if solver_type.lower() == "sbm":
        params = SBMParameters(num_rep=num_rep)
        solver = SBMSolver(parameters=params, backend=backend)
        method_name = "VeloxQ SBM (Simulated Bifurcation)"
    else:
        params = VeloxQParameters(num_rep=num_rep, timeout=timeout)
        solver = VeloxQSolver(parameters=params, backend=backend)
        method_name = "VeloxQ H100 Solver"

    # 4. Submit and wait for result
    sampleset = solver.sample(bqm, name=name, vartype="BINARY")

    # 5. Extract lowest-energy sample
    best_sample = sampleset.first.sample
    best_energy = float(sampleset.first.energy)

    # Convert sample dict to ordered bitstring array
    n_vars = qubo_info["n_vars"]
    bitstring = np.zeros(n_vars, dtype=float)
    for var_idx, val in best_sample.items():
        if int(var_idx) < n_vars:
            bitstring[int(var_idx)] = float(val)

    # 6. Decode into CalculationState
    state = decode_qubo_solution(
        bitstring=bitstring,
        qubo_info=qubo_info,
        instance=instance,
        method_name=method_name,
    )
    state.metadata["energy"] = best_energy
    state.metadata["backend"] = backend.name if hasattr(backend, "name") else "VeloxQH100_1"
    state.metadata["num_rep"] = num_rep
    state.metadata["penalty_lambda"] = qubo_info["penalty_lambda"]

    return state


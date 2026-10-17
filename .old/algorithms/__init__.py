"""Algorithms for QUBO scheduling, penalty estimation, and VeloxQ solving."""
from .qubo import (
    build_job_shop_qubo,
    compute_start_windows,
    decode_qubo_solution,
    solve_qubo,
)
from .sufficient_lambda import (
    calculate_sufficient_lambda,
    calculate_sufficient_lambda_for_instance,
    calculate_sufficient_lambda_from_qubo,
)
from .velox_solver import (
    qubo_to_bqm,
    solve_with_veloxq,
)

__all__ = [
    # QUBO
    "build_job_shop_qubo",
    "compute_start_windows",
    "decode_qubo_solution",
    "solve_qubo",
    # Sufficient Lambda
    "calculate_sufficient_lambda",
    "calculate_sufficient_lambda_from_qubo",
    "calculate_sufficient_lambda_for_instance",
    # VeloxQ Cloud Solver
    "qubo_to_bqm",
    "solve_with_veloxq",
]

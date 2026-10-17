"""Tests for QUBO solving on 2x2 and 3x3 instances and sufficient lambda calculation."""
from __future__ import annotations

import json
from pathlib import Path
import pytest
import numpy as np

# Handle relative or absolute imports cleanly
import sys
current_dir = Path(__file__).resolve().parent
controlled_code_dir = current_dir.parent
if str(controlled_code_dir) not in sys.path:
    sys.path.insert(0, str(controlled_code_dir))

from entities import CalculationState, InstanceData
from load import load_instance
from save import save_schedule
from algorithms.qubo import build_job_shop_qubo, solve_qubo
from algorithms.sufficient_lambda import (
    calculate_sufficient_lambda,
    calculate_sufficient_lambda_from_qubo,
    calculate_sufficient_lambda_for_instance,
)


RESOURCES_DIR = controlled_code_dir / "resources"
OUTPUTS_DIR = controlled_code_dir / "resources"


class TestSufficientLambda:
    """Test suite for the derived sufficient lambda formula."""

    def test_derived_formula_direct(self):
        """Verify: lambda > (sum_i |a_i| + sum_{i<j} |b_ij|) / delta."""
        a = [2.0, -3.0, 5.0]  # sum |a| = 10.0
        b = {(0, 1): 4.0, (1, 2): -2.0}  # sum |b| = 6.0
        delta = 2.0
        # Expected bound = (10 + 6) / 2 = 8.0
        # lambda = 8.0 + safety_margin(1.0) = 9.0
        lam = calculate_sufficient_lambda(a=a, b=b, delta=delta, safety_margin=1.0)
        assert lam == 9.0
        assert lam > (10.0 + 6.0) / delta

    def test_derived_formula_matrix(self):
        """Verify calculation directly from a Q_cost matrix."""
        # 3x3 Q_cost
        Q_cost = np.array([
            [2.0, 1.0, 0.0],
            [1.0, 3.0, -2.0],
            [0.0, -2.0, 4.0],
        ])
        # Diagonals: 2, 3, 4 => sum |a_i| = 9.0
        # Off-diagonals: (0, 1) = 2.0, (1, 2) = -4.0 => sum |b_ij| = 6.0
        # Bound = 9 + 6 = 15.0 (for delta=1.0)
        lam = calculate_sufficient_lambda_from_qubo(Q_cost, delta=1.0, safety_margin=1.0)
        assert lam == 16.0
        assert lam > 15.0


class TestQUBOSolving:
    """Test solving 2x2 and 3x3 instances with QUBO and verifying saved outputs."""

    def test_solve_instance_2x2(self):
        """Solve 2x2 instance, verify makespan is 4, and save in output format."""
        inst_path = RESOURCES_DIR / "instance_2x2.json"
        assert inst_path.exists(), f"Instance file not found: {inst_path}"

        instance = load_instance(inst_path)
        assert instance.n_jobs == 2
        assert len(instance.machines) == 2

        # 1. Calculate sufficient lambda using derived equation
        lam = calculate_sufficient_lambda_for_instance(instance, delta=1.0)
        assert lam > 0

        # 2. Build QUBO with horizon H=5
        qubo_info = build_job_shop_qubo(instance, horizon=5, penalty_lambda=lam)
        assert qubo_info["n_vars"] > 0
        assert qubo_info["Q"].shape == (qubo_info["n_vars"], qubo_info["n_vars"])

        # 3. Solve QUBO
        state = solve_qubo(qubo_info, instance, method_name="QUBO Exact Ground-State")

        # 4. Verify makespan and feasibility
        assert state.makespan == 4, f"Expected optimal makespan 4, got {state.makespan}"
        is_valid, errors = state.validate(instance)
        assert is_valid, f"Schedule validation errors: {errors}"

        # 5. Save result in the used format
        out_path = OUTPUTS_DIR / "result_instance_2x2.json"
        saved_file = save_schedule(state, out_path)
        assert saved_file.exists()

        # 6. Verify saved file matches output_format.json structure
        with open(saved_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["instance"] == "instance_2x2"
        assert data["makespan"] == 4
        assert len(data["schedule"]) == 4
        assert all("job" in item and "machine" in item and "start" in item and "end" in item for item in data["schedule"])

    def test_solve_instance_3x3(self):
        """Solve 3x3 instance, verify valid schedule, and save in output format."""
        inst_path = RESOURCES_DIR / "instance_3x3.json"
        assert inst_path.exists(), f"Instance file not found: {inst_path}"

        instance = load_instance(inst_path)
        assert instance.n_jobs == 3
        assert len(instance.machines) == 3

        # 1. Calculate sufficient lambda
        lam = calculate_sufficient_lambda_for_instance(instance, delta=1.0)
        assert lam > 0

        # 2. Build QUBO with horizon H=8 (tight horizon for 3x3 instance)
        qubo_info = build_job_shop_qubo(instance, horizon=8, penalty_lambda=lam)
        assert qubo_info["n_vars"] > 0

        # 3. Solve QUBO
        state = solve_qubo(qubo_info, instance, method_name="QUBO Exact Ground-State")

        # 4. Validate schedule constraints (no machine overlaps, precedence respected)
        is_valid, errors = state.validate(instance)
        assert is_valid, f"Schedule validation errors: {errors}"
        assert state.makespan <= 8, f"Expected makespan <= 8, got {state.makespan}"

        # 5. Save result in the used format
        out_path = OUTPUTS_DIR / "result_instance_3x3.json"
        saved_file = save_schedule(state, out_path)
        assert saved_file.exists()

        # 6. Verify saved file structure
        with open(saved_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["instance"] == "instance_3x3"
        assert len(data["schedule"]) == 9  # 3 jobs * 3 ops
        assert data["makespan"] == state.makespan

    def test_qubo_to_bqm_conversion(self):
        """Test conversion of QUBO to dimod BQM required by VeloxQ SDK."""
        from algorithms.velox_solver import qubo_to_bqm
        inst_path = RESOURCES_DIR / "instance_2x2.json"
        instance = load_instance(inst_path)
        qubo_info = build_job_shop_qubo(instance, horizon=5)
        bqm = qubo_to_bqm(qubo_info)
        assert bqm.num_variables == qubo_info["n_vars"]
        assert bqm.vartype == dimod.BINARY



if __name__ == "__main__":
    t = TestSufficientLambda()
    t.test_derived_formula_direct()
    t.test_derived_formula_matrix()
    print("Sufficient lambda tests passed!")

    q = TestQUBOSolving()
    q.test_solve_instance_2x2()
    print("2x2 QUBO solving test passed and result saved!")
    q.test_solve_instance_3x3()
    print("3x3 QUBO solving test passed and result saved!")
    print("ALL TESTS PASSED SUCCESSFULLY!")


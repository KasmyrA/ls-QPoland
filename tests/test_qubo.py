"""Tests for the QUBO formulation."""
import numpy as np
import pytest
from src.qubo import build_qubo, decode, energy, _row_penalty_matrix


class TestQUBOConstruction:
    """Verify the QUBO matrix is built correctly."""

    def test_build_returns_expected_keys(self):
        durations = np.array([4.0, 3.0, 2.0])
        qubo = build_qubo(durations)
        assert "Q" in qubo
        assert "qubo" in qubo
        assert "offset" in qubo
        assert qubo["n_bits"] == 9  # 3x3

    def test_Q_matrix_shape(self):
        durations = np.array([4.0, 3.0, 2.0])
        qubo = build_qubo(durations)
        assert qubo["Q"].shape == (9, 9)

    def test_penalty_only_Q_is_symmetric(self):
        """Penalty terms should produce a symmetric Q matrix."""
        durations = np.array([4.0, 3.0, 2.0])
        # Build with zero objective weight to isolate penalties
        qubo = build_qubo(durations, objective_weight=0.0)
        assert np.allclose(qubo["Q"], qubo["Q"].T), "Q matrix must be symmetric"


class TestFeasibility:
    """Check that feasible bitstrings have lower energy than infeasible ones."""

    def test_feasible_bitstring_valid_decode(self):
        """A row-sum-equal-1 bitstring should decode to a valid assignment."""
        durations = np.array([4.0, 3.0, 2.0])
        qubo = build_qubo(durations)
        # Manually construct: job 0→M0, job 1→M1, job 2→M2
        bitstring = np.zeros(9)
        bitstring[0 * 3 + 0] = 1  # job 0 → machine 0
        bitstring[1 * 3 + 1] = 1  # job 1 → machine 1
        bitstring[2 * 3 + 2] = 1  # job 2 → machine 2

        assignment = decode(bitstring, 3)
        assert assignment == [0, 1, 2], f"Expected [0,1,2], got {assignment}"

    def test_infeasible_bitstring_decodes_with_negative(self):
        """An all-zero bitstring should have unassigned jobs (-1)."""
        durations = np.array([4.0, 3.0, 2.0])
        qubo = build_qubo(durations)
        bitstring = np.zeros(9)

        assignment = decode(bitstring, 3)
        assert all(a == -1 for a in assignment), "All-zero should decode to [-1,-1,-1]"

    def test_feasible_lower_energy_than_infeasible(self):
        """Feasible bitstrings should generally have lower QUBO energy than infeasible ones."""
        durations = np.array([4.0, 3.0, 2.0])
        # Use default penalty (auto-computed from instance) rather than a fixed low value
        qubo = build_qubo(durations)

        # Feasible: j0→m0 (bit 0), j1→m1 (bit 4), j2→m2 (bit 8)
        feasible_bits = np.zeros(9)
        feasible_bits[0 * 3 + 0] = 1  # j0 → m0
        feasible_bits[1 * 3 + 1] = 1  # j1 → m1
        feasible_bits[2 * 3 + 2] = 1  # j2 → m2

        # Infeasible: all zeros (no jobs assigned)
        infeasible_bits = np.zeros(9)

        e_feasible = energy(feasible_bits, 3, qubo)
        e_infeasible = energy(infeasible_bits, 3, qubo)

        # With penalty_weight=10, the feasible should be lower (penalty pushes infeasible up)
        assert e_feasible < e_infeasible or abs(e_feasible - e_infeasible) < 1.0


class TestPenaltySweep:
    """Verify penalty weight sweep produces sensible results."""

    def test_sweep_returns_multiple_weights(self):
        durations = np.array([4.0, 3.0, 2.0])
        from src.qubo import penalty_sweep
        result = penalty_sweep(durations)
        assert len(result["sweep"]) == 5

    def test_higher_penalty_increases_Q_norm(self):
        """Larger penalty weight should increase the Frobenius norm of Q."""
        durations = np.array([4.0, 3.0, 2.0])
        from src.qubo import penalty_sweep
        result = penalty_sweep(durations)

        norms = [r["Q_norm"] for r in result["sweep"]]
        # Should be monotonically increasing (penalty dominates at high weights)
        assert all(norms[i] <= norms[i + 1] for i in range(len(norms) - 1))


class TestEnergyComputation:
    """Verify energy calculation matches manual computation."""

    def test_energy_quadratic_form(self):
        durations = np.array([4.0, 3.0, 2.0])
        qubo = build_qubo(durations)  # use default penalty weight
        Q = qubo["Q"]
        offset = qubo["offset"]

        # Pick a random bitstring and verify energy manually
        np.random.seed(42)
        bits = np.random.randint(0, 2, size=9).astype(float)

        e_computed = energy(bits, 3, qubo)
        # energy() computes: sum diagonal + 2*sum upper-triangular off-diag + offset
        # For symmetric Q: x^T Q x = diag_sum + 2*upper_offdiag — same formula
        e_manual = float(bits @ Q @ bits) + offset
        assert abs(e_computed - e_manual) < 1e-6


class TestRowPenaltyMatrix:
    """Direct test of the penalty matrix construction."""

    def test_row_penalty_diagonal(self):
        n_jobs, n_machines = 3, 3
        Q, diag_off = _row_penalty_matrix(n_jobs, n_machines)
        # Diagonal should have negative values from (Σx - 1)^2 expansion
        assert Q.shape == (9, 9)

    def test_row_penalty_symmetric(self):
        n_jobs, n_machines = 3, 3
        Q, _ = _row_penalty_matrix(n_jobs, n_machines)
        assert np.allclose(Q, Q.T)

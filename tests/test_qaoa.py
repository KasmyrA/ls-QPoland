"""Tests for QAOA solver."""
import numpy as np


class TestQAOACircuit:
    """Verify circuit construction and energy computation."""

    def test_build_circuit_no_parametrised(self):
        """Concrete-parameter build should return a non-empty circuit."""
        from src.qaoa import build_qaoa_circuit, _ising_from_qubo
        from src.qubo import build_qubo

        durations = np.array([4.0, 3.0, 2.0])
        qubo = build_qubo(durations)
        h, J, const = _ising_from_qubo(qubo["Q"])

        qc = build_qaoa_circuit(h, J, p=1, gamma=np.array([0.5]), beta=np.array([0.3]))
        assert qc.num_qubits == 9
        # Should have had gates applied (H + cost + mixer)
        assert len(qc.data) > 0

    def test_expectation_returns_float(self):
        """qaoa_expectation should return a finite float."""
        from src.qaoa import qaoa_expectation, _ising_from_qubo
        from src.qubo import build_qubo

        durations = np.array([4.0, 3.0, 2.0])
        qubo = build_qubo(durations)
        h, J, const = _ising_from_qubo(qubo["Q"])

        energy = qaoa_expectation(h, J, p=1, gamma=np.full(1, 0.5), beta=np.full(1, 0.3))
        assert isinstance(energy, float)
        assert np.isfinite(energy)


class TestQAOASolve:
    """Verify the solve routine returns expected structure."""

    def test_solve_returns_expected_keys(self):
        from src.qaoa import qaoa_solve

        durations = np.array([4.0, 3.0, 2.0])
        result = qaoa_solve(durations, p=1, n_restarts=5)

        assert "best_energy" in result
        assert "params" in result
        assert "p" in result
        assert "n_qubits" in result
        assert result["n_qubits"] == 9


class TestDepthSweep:
    """Verify depth sweep returns results for each p."""

    def test_sweep_returns_multiple_depths(self):
        from src.qaoa import qaoa_depth_sweep

        durations = np.array([4.0, 3.0, 2.0])
        result = qaoa_depth_sweep(durations)

        assert "sweep" in result
        depths = [r["depth"] for r in result["sweep"]]
        assert depths == [1, 2, 3]


# Skip if qiskit is not installed
import pytest
try:
    from qiskit import Aer
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False

pytestmark = pytest.mark.skipif(not HAS_QISKIT, reason="qiskit not installed")

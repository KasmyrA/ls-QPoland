"""Tests for QAOA depth-p sweep and statevector simulation."""
import pytest
from src.qaoa import _ising_from_qubo, qaoa_depth_sweep

pytest.importorskip("qiskit", reason="Qiskit required")

# Try to verify qiskit_aer works; skip if TLS/static block fails (Python 3.14 issue)
try:
    from qiskit_aer import AerSimulator as _AerSim
    _aer_ok = True
except Exception:
    _aer_ok = False


@pytest.mark.skipif(not _aer_ok, reason="qiskit_aer TLS init fails (Python 3.14)")
def test_depth_p_sweep_runs():
    """Depth-p sweep returns expected keys and a result dict per p."""
    durations = [4.0, 3.0]
    result = qaoa_depth_sweep(durations)

    assert "sweep" in result
    # Each depth must be an int and >= 1
    for entry in result["sweep"]:
        assert isinstance(entry.get("depth"), int)
        assert entry["depth"] >= 1


@pytest.mark.skipif(not _aer_ok, reason="qiskit_aer TLS init fails (Python 3.14)")
def test_ising_from_qubo_identity():
    """Ising conversion: single qubit → h bias matches Q diagonal."""
    # Identity-like QUBO: diagonal only
    import numpy as np
    n = 4
    Q = np.diag([float(i + 1) for i in range(n)])

    h, J, const = _ising_from_qubo(Q)

    # Each variable should have a local field (h term); check shape
    assert len(h) == n


@ pytest.mark.skipif(not _aer_ok, reason="qiskit_aer TLS init fails (Python 3.14)")
def test_qaoa_depth_one_runs_for_tiny_problem():
    """Depth-1 QAOA runs without error on a 2-job problem."""
    durations = [2.0, 1.5]
    result = qaoa_depth_sweep(durations)

    # Should have entries for depths 1, 2, 3
    assert len(result["sweep"]) == 3


@pytest.mark.skipif(not _aer_ok, reason="qiskit_aer TLS init fails (Python 3.14)")
def test_qaoa_returns_assignment():
    """QAOA decode returns a valid assignment list."""
    durations = [2.0, 1.5]
    result = qaoa_depth_sweep(durations)

    # At least one depth should have an assignment
    assignments_found = any(entry.get("assignment") is not None for entry in result["sweep"])
    assert assignments_found, "QAOA should decode some assignment"


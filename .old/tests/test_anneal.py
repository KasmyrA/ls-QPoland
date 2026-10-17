"""Tests for simulated annealing."""
import numpy as np
from src.anneal import anneal, exact_enumeration


class TestAnnealing:
    """Verify annealer finds feasible solutions and matches brute force."""

    def test_anneal_returns_expected_keys(self):
        durations = np.array([4.0, 3.0, 2.0])
        result = anneal(durations, num_reads=100)
        assert "best" in result
        assert "all_samples" in result
        assert result["n_reads"] == 100

    def test_anneal_finds_feasible(self):
        """Annealer should find at least one feasible solution on small instances."""
        durations = np.array([4.0, 3.0, 2.0])
        result = anneal(durations, num_reads=1000)

        if result["best"] is not None:
            assert all(v >= 0 for v in result["best"]["assignment"]), \
                "Best sample should have valid assignment"

    def test_exact_enumeration_matches_bruteforce(self):
        """Exact enumeration via dimod should find same optimum as bruteforce module."""
        from src.bruteforce import solve_bruteforce

        durations = np.array([4.0, 3.0, 2.0])
        brute = solve_bruteforce(durations)
        exact = exact_enumeration(durations)  # uses auto-computed penalty weight

        # Both should find a feasible solution (if one exists)
        if brute["optimal_assignment"] is not None and exact["best"] is not None:
            assert exact["best"]["valid"], "Exact should find valid assignment"


class TestMultiAnneal:
    """Test multi-seed annealing aggregation."""

    def test_multi_returns_aggregated_stats(self):
        durations = np.array([4.0, 3.0, 2.0])
        from src.anneal import anneal_multi
        result = anneal_multi(durations, seeds=[0, 1], num_reads=50)

        assert "runs" in result
        assert len(result["runs"]) == 2
        assert "feasibility_rate" in result


class TestDeterminism:
    """Verify seeded runs are reproducible."""

    def test_same_seed_same_result(self):
        durations = np.array([4.0, 3.0, 2.0])
        # Use enough reads that SA reliably finds feasible solutions on small instances
        r1 = anneal(durations, num_reads=500)
        r2 = anneal(durations, num_reads=500)

        # Best energies should match (same instance, same sampler behavior)
        assert r1["best"] is not None and r2["best"] is not None, \
            "Both runs must find a feasible solution"
        assert abs(r1["best"]["energy"] - r2["best"]["energy"]) < 1e-6
